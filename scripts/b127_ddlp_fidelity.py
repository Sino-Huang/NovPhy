"""Issue #135 G-fid Tier E: the official DDLP PHYRE-128 video-prediction evaluation with the released checkpoint.

Runs taldatech/ddlp's unchanged ``eval/eval_gen_metrics.py`` (from the read-only clone) on the released PHYRE-128
checkpoint (``ddlp-phyre/phyre_ddlp.pth`` + ``hparams.json``) and the authors' released PHYRE test split (400 episodes),
condition 10 frames, predict 90 (horizon 100; DDLP paper Table 3 / section 5.2), deterministic (the script's default),
on the CPU (env ``b127-ddlp``: torch 2.13, piqa 1.3.1). The band is frozen in ``fidelity/band.json`` and posted on
#135 before any run; this script refuses to run without it and records its sha256.

Two harness facts, neither an edit of the clone:
* ``hparams.json`` names the authors' data root (``/media/newhd/data/phyre/full``); the run directory holds a copy of
  the released ``hparams.json`` whose only change is ``root`` -> the extracted release data
  (``data/external/ddlp/data/phyre``). The checkpoint is passed with ``--checkpoint``.
* ``datasets/get_dataset.py`` imports ``datasets.langtable_ds``, which does not exist at any commit of the repo; an
  empty stand-in module with the two imported names is registered before the unchanged script runs (the PHYRE path
  never uses it).

Modes (``b127-ddlp`` env, CPU):
    --prepare   extract the test split, write the run directory (hparams copy), verify every input against MANIFEST
    --run       one attempt: the official script in a child process (log in fidelity/), then fidelity/gfid.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DDLP_REF = Path(os.environ.get("DDLP_REF", "/mnt/array/sukaih/novphy-baselines-src/ddlp@5686275"))
COMMIT = "5686275800ee58d9832714af5ed8363c4fb606bc"
OUT = ROOT / ".local-artifacts/issue-135-ddlp-v1"
FID = OUT / "fidelity"
RUN = FID / "run"
EXTERNAL = ROOT / "data/external/ddlp"
CKPT_DIR = EXTERNAL / "checkpoints/ddlp-phyre"
DATA_ZIP = EXTERNAL / "data/phyre_dataset.zip"
DATA_ROOT = EXTERNAL / "data/phyre"
SCRIPT = DDLP_REF / "eval/eval_gen_metrics.py"
COND_STEPS, HORIZON, BATCH = 10, 100, 2
THREADS = int(os.environ.get("B135_CPU_THREADS", "4"))


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def frozen_band() -> tuple[dict, str]:
    path = FID / "band.json"
    if not path.is_file():
        raise SystemExit("fidelity/band.json missing: freeze and post the band (Decision P-fid) before any run")
    return json.loads(path.read_text()), sha256(path)


def manifest_entry(name: str) -> dict:
    for sub in ("checkpoints", "data"):
        path = EXTERNAL / sub / "MANIFEST.json"
        for entry in json.loads(path.read_text())["files"]:
            if entry["file"] == name:
                return entry
    raise KeyError(name)


def verify_downloads() -> dict:
    out = {}
    for name, path in (("ddlp-phyre.zip", EXTERNAL / "checkpoints/ddlp-phyre.zip"), ("phyre_dataset.zip", DATA_ZIP)):
        entry = manifest_entry(name)
        if not entry.get("meta_mac_ok") or sha256(path) != entry["sha256"]:
            raise ValueError(f"{name} differs from data/external/ddlp MANIFEST")
        out[name] = entry["sha256"]
    with zipfile.ZipFile(EXTERNAL / "checkpoints/ddlp-phyre.zip") as z:
        for member in ("ddlp-phyre/phyre_ddlp.pth", "ddlp-phyre/hparams.json"):
            extracted = EXTERNAL / "checkpoints" / member
            if hashlib.sha256(z.read(member)).hexdigest() != sha256(extracted).split(":")[1]:
                raise ValueError(f"{extracted} differs from its release zip member")
            out[member] = sha256(extracted)
    return out


def test_split_digest() -> str:
    """sha256 over (relative path, file sha256) of every extracted test file, sorted."""
    digest = hashlib.sha256()
    base = DATA_ROOT / "test"
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        digest.update(f"{path.relative_to(DATA_ROOT)}\t{sha256(path)}\n".encode())
    return "sha256:" + digest.hexdigest()


def prepare() -> int:
    if not (DATA_ROOT / "test").is_dir():
        with zipfile.ZipFile(DATA_ZIP) as z:
            z.extractall(DATA_ROOT, [m for m in z.namelist() if m.startswith("test/")])
    with zipfile.ZipFile(DATA_ZIP) as z:
        names = {m for m in z.namelist() if m.startswith("test/") and not m.endswith("/")}
    on_disk = {str(p.relative_to(DATA_ROOT)) for p in (DATA_ROOT / "test").rglob("*") if p.is_file()}
    if names != on_disk:
        raise ValueError("extracted test split differs from the release zip")
    run_ckpt = RUN / "ddlp-phyre"
    run_ckpt.mkdir(parents=True, exist_ok=True)
    hparams = json.loads((CKPT_DIR / "hparams.json").read_text())
    released_root = hparams["root"]
    hparams["root"] = str(DATA_ROOT)
    (run_ckpt / "hparams.json").write_text(json.dumps(hparams, indent=2) + "\n")
    write_json(FID / "prepare.json", {
        "schema": "issue_135_gfid_prepare_v1", "written_at": utc_now(), "inputs": verify_downloads(),
        "test_episodes": sum(1 for p in (DATA_ROOT / "test").iterdir() if p.is_dir()),
        "test_split_digest": test_split_digest(),
        "hparams_override": {"root": {"released": released_root, "run": str(DATA_ROOT)}},
        "run_hparams_sha256": sha256(run_ckpt / "hparams.json")})
    print(json.dumps(json.loads((FID / "prepare.json").read_text()), indent=1))
    return 0


def official_command(batch: int = BATCH) -> list[str]:
    return ["-d", "phyre", "-p", str(RUN / "ddlp-phyre"), "--checkpoint", str(CKPT_DIR / "phyre_ddlp.pth"),
            "-c", str(COND_STEPS), "--horizon", str(HORIZON), "-b", str(batch), "--cpu"]


def exec_official(argv: list[str]) -> None:
    """Child process: register the missing-module stand-in, then run the unchanged official script as __main__."""
    import runpy
    import types
    import torch
    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(DDLP_REF))
    stub = types.ModuleType("datasets.langtable_ds")
    stub.LanguageTableDataset = stub.LanguageTableDatasetImage = None
    sys.modules["datasets.langtable_ds"] = stub
    sys.argv = [str(SCRIPT)] + argv
    runpy.run_path(str(SCRIPT), run_name="__main__")


def exec_diagnose(argv: list[str]) -> None:
    """Child process of a diagnostic: the official script, unchanged, on the first ``argv[0]`` test episodes (the
    official dataset wrapped in ``Subset``); ``argv[1:]`` are the official arguments."""
    import runpy
    import types
    import torch
    from torch.utils.data import Subset
    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(DDLP_REF))
    stub = types.ModuleType("datasets.langtable_ds")
    stub.LanguageTableDataset = stub.LanguageTableDatasetImage = None
    sys.modules["datasets.langtable_ds"] = stub
    import datasets.get_dataset as gd
    n, official = int(argv[0]), argv[1:]
    original = gd.get_video_dataset

    def first_n(*args, **kwargs):
        ds = original(*args, **kwargs)
        return Subset(ds, range(min(n, len(ds))))

    gd.get_video_dataset = first_n
    sys.argv = [str(SCRIPT)] + official
    runpy.run_path(str(SCRIPT), run_name="__main__")


def env_versions(python: str) -> dict:
    code = ("import json, torch, torchvision, numpy, piqa, PIL; print(json.dumps({'torch': torch.__version__, "
            "'torchvision': torchvision.__version__, 'numpy': numpy.__version__, 'piqa': piqa.__version__, "
            "'pillow': PIL.__version__}))")
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    return json.loads(out.stdout.strip().splitlines()[-1])


def diagnose(tag: str, episodes: int, batch: int) -> int:
    """A pre-registered diagnostic (Decision P-fid issuecomment-6397579863): the official script on the first
    ``episodes`` test episodes at ``batch`` in the interpreter running this script; fidelity/diagnostics/<tag>.json."""
    prep = json.loads((FID / "prepare.json").read_text())
    run_dir = FID / "diagnostics" / f"run-{tag}" / "ddlp-phyre"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "hparams.json").write_bytes((RUN / "ddlp-phyre/hparams.json").read_bytes())
    metrics_path = run_dir / "eval/last_val_image_metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()
    official = ["-d", "phyre", "-p", str(run_dir), "--checkpoint", str(CKPT_DIR / "phyre_ddlp.pth"), "-c",
                str(COND_STEPS), "--horizon", str(HORIZON), "-b", str(batch), "--cpu"]
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--exec-diagnose", str(episodes)] + official
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(THREADS))
    log_path = FID / "diagnostics" / f"{tag}.log"
    began = time.monotonic()
    with open(log_path, "w") as log:
        code = subprocess.call(cmd, cwd=str(DDLP_REF), stdout=log, stderr=subprocess.STDOUT, env=env)
    metrics = json.loads(metrics_path.read_text()) if code == 0 and metrics_path.is_file() else None
    record = {"schema": "issue_135_gfid_diagnostic_v1", "tag": tag, "episodes": episodes, "batch": batch,
              "python": sys.executable, "versions": env_versions(sys.executable), "exit_code": code,
              "metrics": metrics, "official_args": official, "inputs": prep["inputs"],
              "wall_seconds": time.monotonic() - began, "log": str(log_path.relative_to(ROOT)),
              "log_sha256": sha256(log_path), "decision": "https://github.com/Sino-Huang/NovPhy/issues/135"
                                                          "#issuecomment-6397579863", "written_at": utc_now()}
    write_json(FID / "diagnostics" / f"{tag}.json", record)
    print(json.dumps({"tag": tag, "metrics": metrics, "versions": record["versions"]}))
    return 0 if code == 0 else 1


def torch_hub_inputs() -> dict:
    hub = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch")) / "hub/checkpoints"
    return {name: sha256(hub / name) for name in ("vgg16-397923af.pth", "vgg.pth") if (hub / name).is_file()}


def run(repair_decision: str | None = None, batch: int = BATCH) -> int:
    band, band_digest = frozen_band()
    prep = json.loads((FID / "prepare.json").read_text())
    inputs = verify_downloads()
    if inputs != prep["inputs"] or sha256(RUN / "ddlp-phyre/hparams.json") != prep["run_hparams_sha256"]:
        raise ValueError("inputs changed since --prepare")
    gfid_path = FID / "gfid.json"
    previous = json.loads(gfid_path.read_text()) if gfid_path.is_file() else None
    attempts = list(previous["attempts"]) if previous else []
    if previous and previous.get("complete") and (previous.get("pass") or repair_decision is None):
        raise SystemExit("gfid.json is complete; a repair attempt needs --repair-decision <posted P-fid URL>")
    n = len(attempts) + 1
    if n > 3:
        raise SystemExit("P-fid: at most two repair attempts")
    log_path = FID / f"eval.attempt{n}.log"
    metrics_path = RUN / "ddlp-phyre/eval/last_val_image_metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()
    command = official_command(batch)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--exec-official", "--"] + command
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(THREADS))
    started, began = utc_now(), time.monotonic()
    with open(log_path, "w") as log:
        code = subprocess.call(cmd, cwd=str(DDLP_REF), stdout=log, stderr=subprocess.STDOUT, env=env)
    wall = time.monotonic() - began
    attempts.append({"attempt": n, "started_at": started, "ended_at": utc_now(), "exit_code": code,
                     "wall_seconds": wall, "log": str(log_path.relative_to(ROOT)), "log_sha256": sha256(log_path),
                     "python": sys.executable, "versions": env_versions(sys.executable), "batch": batch,
                     "repair_decision": repair_decision})
    metrics = json.loads(metrics_path.read_text()) if code == 0 and metrics_path.is_file() else None
    tier = band["tiers"]["tier-e"]
    value = None if metrics is None else float(metrics[tier["statistic_key"]])
    lo, hi = tier["band"]
    in_band = value is not None and lo <= value <= hi
    descriptive = {}
    if metrics is not None:
        for key, d in tier["descriptive"].items():
            v = float(metrics[key])
            descriptive[key] = {"ours": v, "paper": d["paper"], "band": d["band"],
                                "in_band": d["band"][0] <= v <= d["band"][1]}
    record = {
        "schema": "issue_127_gfid_v1", "issue": 135, "arm": "T2 DDLP", "commit": COMMIT,
        "band_sha256": band_digest, "rules": band["rule"], "complete": metrics is not None,
        "command": ["python", str(SCRIPT.relative_to(DDLP_REF))] + command,
        "cwd": str(DDLP_REF), "device": "cpu", "threads": THREADS,
        "harness": {"langtable_ds_stand_in": "empty module registered as datasets.langtable_ds (missing upstream)",
                    "hparams_override": prep["hparams_override"]},
        "inputs": {**inputs, "test_split_digest": prep["test_split_digest"], "torch_hub": torch_hub_inputs()},
        "env_freeze_sha256": sha256(OUT / "envs/b127-ddlp.pip-freeze.txt"),
        "attempts": attempts,
        "tiers": {"tier-e": {"metric": tier["metric"], "paper": tier["paper"], "band": tier["band"], "ours": value,
                             "in_band": in_band, "official_results": metrics, "descriptive": descriptive}},
        "compute": {"tier-e": {"cpu_wall_seconds": sum(a["wall_seconds"] for a in attempts), "gpu_seconds": 0.0,
                               "attempts": len(attempts)}},
        "G_fid": bool(in_band), "pass": bool(in_band), "written_at": utc_now(),
    }
    write_json(gfid_path, record)
    print(json.dumps({"ours": value, "band": tier["band"], "pass": in_band, "descriptive": descriptive}))
    return 0 if code == 0 else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--exec-official":
        exec_official(argv[2:] if len(argv) > 1 and argv[1] == "--" else argv[1:])
        return 0
    if argv and argv[0] == "--exec-diagnose":
        exec_diagnose(argv[1:])
        return 0
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--diagnose", metavar="TAG")
    p.add_argument("--episodes", type=int, default=50, help="--diagnose: first N test episodes")
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--repair-decision", default=None, help="--run after a failed attempt: the posted P-fid URL")
    a = p.parse_args(argv)
    if a.prepare:
        return prepare()
    if a.diagnose:
        return diagnose(a.diagnose, a.episodes, a.batch)
    return run(a.repair_decision, a.batch)


if __name__ == "__main__":
    sys.exit(main())
