"""#136 G-fid (Decision P-fid on #136): Diffusion Forcing, Table 2 time series, Exchange test CRPS-sum, Tier T, with
the vendored official pipeline (``world_model/baselines/diffusion_forcing/vendor``, buoyancy99/diffusion-forcing
``paper`` @ dd65516) in env ``b127-dforce``.

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template
[repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in
`README.md` and the `LICENSE` file to credit the author. (Upstream fork-credit sentence, kept as upstream modules do.)

Modes:
- ``--data``: download GluonTS ``exchange_rate_nips`` (the tarball GluonTS 0.13.1 fetches) to
  ``data/external/diffusion_forcing/`` (``MANIFEST.json``: url, bytes, sha256, downloaded_at) and materialise it with
  ``gluonts.dataset.repository.datasets.get_dataset`` under ``data/external/diffusion_forcing/gluonts-datasets``.
- ``--run [--device cpu|cuda] [--seed s]``: the official run, resumable. Phase 1 = ``python vendor/main.py
  +name=gfid_exchange_s<seed> experiment=exp_prediction dataset=ts_exchange algorithm=df_prediction
  experiment.tasks=[save_mean_std_metadata,training]`` (early stopping on validation CRPS-sum; a ``last.ckpt`` at every
  validation via ``+experiment.training.checkpointing.save_last=true``; a resumed phase passes ``load=last.ckpt``);
  phase 2 = ``experiment.tasks=[test] load=<best checkpoint>`` (the official test task on the best checkpoint, as the
  official ``[training, test]`` sequence does) whose printed ``test/crps_sum`` is the G-fid number. Run-time settings
  that do not touch the method: ``wandb.mode=disabled``, ``lightning.seed_everything(seed)`` before ``main`` (upstream
  sets no seed), GluonTS' ``$HOME/.gluonts`` pointed at the data directory (a HOME sandbox), fp32 on a CPU (Lightning has
  no CPU fp16-mixed; the official 16-mixed on a GPU). SIGTERM/SIGUSR1 -> the child is stopped -> exit 75 (resume from
  the last validation's ``last.ckpt``; RNG streams after a resume differ from an uninterrupted run).
- ``--collect``: ``fidelity/gfid.json`` (schema ``issue_127_gfid_v1``) from the run's result and the frozen
  ``fidelity/band.json`` (pass iff low <= crps_sum < high).
- ``--official SEED -- <hydra overrides>`` (internal): the child process (seed, then ``runpy`` of the vendored main).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "world_model" / "baselines" / "diffusion_forcing" / "vendor"
OUTPUT = ROOT / ".local-artifacts" / "issue-136-diffusion-forcing-v1"
FIDELITY = OUTPUT / "fidelity"
DATA = ROOT / "data" / "external" / "diffusion_forcing"
DATASET = "exchange_rate_nips"
URL = "https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/exchange_rate_nips.tar.gz"
SEED = 20260908
EXIT_PAUSED = 75
DECISION = "https://github.com/Sino-Huang/NovPhy/issues/136#issuecomment-6413164397"
UPSTREAM_COMMIT = "dd65516a470b0eda08dcc0f9cb55c87d6527eb9e"
TEST_METRIC = re.compile(r"test/crps_sum\s*[│|]?\s*([0-9.]+(?:[eE][-+]?[0-9]+)?)")
STOP = []


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def log(message) -> None:
    print(f"[issue-136 gfid] {message}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# --data
# ---------------------------------------------------------------------------------------------------------------

def data(output=OUTPUT) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    tarball = DATA / f"{DATASET}.tar.gz"
    if not tarball.is_file():  # the archive GluonTS' get_dataset fetches (kept for the manifest)
        urllib.request.urlretrieve(URL, tarball)
    gluon = DATA / "gluonts-datasets"
    if not (gluon / DATASET).is_dir():
        from gluonts.dataset.repository.datasets import get_dataset
        get_dataset(DATASET, path=gluon, regenerate=False)
    files = sorted(p for p in (gluon / DATASET).rglob("*") if p.is_file())
    write_json(DATA / "MANIFEST.json", {
        "schema": "issue_136_data_manifest_v1", "dataset": DATASET, "url": URL, "bytes": tarball.stat().st_size,
        "sha256": sha256(tarball), "downloaded_at": utc_now(),
        "materialised": {str(p.relative_to(DATA)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}})
    log(f"data ready: {tarball} ({tarball.stat().st_size} bytes), {len(files)} materialised files")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --run
# ---------------------------------------------------------------------------------------------------------------

def run_dir(seed, output=OUTPUT) -> Path:
    return Path(output) / "fidelity" / "run" / f"seed-{seed}"


def home_sandbox(rdir) -> Path:
    """HOME for the child: ``$HOME/.gluonts/datasets`` -> the materialised data (GluonTS hard-codes HOME)."""
    home = rdir / "home"
    link = home / ".gluonts" / "datasets"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(DATA / "gluonts-datasets", target_is_directory=True)
    return home


def overrides(seed, device, tasks, load=None, rdir=None):
    args = [f"+name=gfid_exchange_s{seed}", "experiment=exp_prediction", "dataset=ts_exchange",
            "algorithm=df_prediction", "wandb.mode=disabled", "wandb.entity=novphy",
            f"experiment.tasks=[{','.join(tasks)}]", "+experiment.training.checkpointing.save_last=true",
            f"hydra.run.dir={rdir / 'hydra'}"]
    if device == "cpu":
        args += ["experiment.training.precision=32", "experiment.validation.precision=32",
                 "experiment.test.precision=32"]
    if load is not None:
        args.append(f"load='{load}'")  # quoted: Lightning checkpoint names contain '='
    return args


def best_checkpoint(rdir):
    ckpts = sorted(p for p in (rdir / "hydra" / "checkpoints").glob("*.ckpt") if p.name != "last.ckpt")
    if len(ckpts) != 1:
        raise RuntimeError(f"expected one best checkpoint (save_top_k 1), found {[p.name for p in ckpts]}")
    return ckpts[0]


def _stop(signum, _frame):
    STOP.append(signum)


def child(seed, args, rdir, device, attempt_log) -> int:
    """Run the official child; a stop signal terminates it (SIGTERM, then SIGKILL after 60 s)."""
    env = dict(os.environ, HOME=str(home_sandbox(rdir)), PYTHONUNBUFFERED="1", WANDB_MODE="disabled")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--official", str(seed), "--", *args]
    with open(attempt_log, "a") as out:
        out.write(f"\n### {utc_now()} {' '.join(command)}\n")
        out.flush()
        proc = subprocess.Popen(command, cwd=rdir, env=env, stdout=out, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if STOP:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return EXIT_PAUSED
            time.sleep(2)
    return proc.returncode


def run(device="cpu", seed=SEED, output=OUTPUT) -> int:
    """Resumable official run (phase 1 training to early stop, phase 2 test on the best checkpoint)."""
    if not (Path(output) / "fidelity" / "band.json").is_file():
        raise SystemExit("fidelity/band.json is not frozen: no fidelity run before Decision P-fid")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGUSR1, _stop)
    rdir = run_dir(seed, output)
    rdir.mkdir(parents=True, exist_ok=True)
    progress_path = rdir / "progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else {
        "seed": seed, "phase": "train", "segments": [], "device": device, "started_at": utc_now()}
    attempt_log = rdir / "run.log"
    while progress["phase"] in ("train", "test"):
        began = time.monotonic()
        if progress["phase"] == "train":
            last = rdir / "hydra" / "checkpoints" / "last.ckpt"
            args = overrides(seed, device, ["save_mean_std_metadata", "training"], last if last.is_file() else None,
                             rdir)
        else:
            args = overrides(seed, device, ["test"], best_checkpoint(rdir), rdir)
        code = child(seed, args, rdir, device, attempt_log)
        seconds = time.monotonic() - began
        progress["segments"].append({"phase": progress["phase"], "device": device, "seconds": seconds,
                                     "exit": code, "ended_at": utc_now()})
        if code == EXIT_PAUSED:
            write_json(progress_path, progress)
            log(f"paused in phase {progress['phase']} (exit 75)")
            return EXIT_PAUSED
        if code != 0:
            write_json(progress_path, progress)
            raise SystemExit(f"official run failed in phase {progress['phase']} (exit {code}); see {attempt_log}")
        if progress["phase"] == "train":
            progress.update(phase="test", best_checkpoint=str(best_checkpoint(rdir).relative_to(rdir)))
        else:
            values = TEST_METRIC.findall(attempt_log.read_text().rsplit("### ", 1)[-1])
            if not values:
                raise SystemExit("test/crps_sum not found in the official test output")
            progress.update(phase="done", crps_sum=float(values[-1]), finished_at=utc_now())
        write_json(progress_path, progress)
    log(f"seed {seed}: test CRPS-sum {progress['crps_sum']}")
    return 0


def official(seed, args) -> int:
    """The child: seed every generator (upstream sets none), then the vendored ``main.py`` under hydra."""
    import lightning
    import runpy
    lightning.seed_everything(seed, workers=True)
    sys.path.insert(0, str(VENDOR))
    sys.argv = [str(VENDOR / "main.py"), *args]
    runpy.run_path(str(VENDOR / "main.py"), run_name="__main__")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --collect
# ---------------------------------------------------------------------------------------------------------------

def collect(output=OUTPUT, seeds=(SEED,)) -> int:
    output = Path(output)
    band_path = output / "fidelity" / "band.json"
    band = read_json(band_path)
    runs, seconds = {}, {"cpu_seconds_wall": 0.0, "gpu_seconds": 0.0}
    for seed in seeds:
        progress = read_json(run_dir(seed, output) / "progress.json")
        runs[str(seed)] = {k: progress.get(k) for k in ("crps_sum", "phase", "best_checkpoint", "segments",
                                                         "started_at", "finished_at")}
        for seg in progress["segments"]:
            seconds["gpu_seconds" if seg["device"].startswith("cuda") else "cpu_seconds_wall"] += seg["seconds"]
    complete = all(r["phase"] == "done" for r in runs.values())
    values = [r["crps_sum"] for r in runs.values() if r["crps_sum"] is not None]
    value = values[0] if len(values) == 1 else None
    passed = bool(complete and value is not None and band["low"] <= value < band["high"])
    import torch
    write_json(output / "fidelity" / "gfid.json", {
        "schema": "issue_127_gfid_v1", "issue": 136, "arm": "T4 DF-car", "tier": "T", "G_fid": passed,
        "pass": passed, "complete": complete, "paper_value": band["reported"]["mean"],
        "band": [band["low"], band["high"]], "band_rule": band["rule"], "band_sha256": sha256(band_path),
        "value": value, "statistic": band["metric"], "environment": band["experiment"], "runs": runs,
        "commit": UPSTREAM_COMMIT, "decision": DECISION, "compute": seconds, "repairs": [],
        "script_sha256": sha256(Path(__file__)), "torch": torch.__version__,
        "deviations": band["code"]["runtime_settings"], "written_at": utc_now(),
        "addendum": "https://github.com/Sino-Huang/NovPhy/issues/136#issuecomment-6413855182"})
    log(f"gfid: value {value}, band [{band['low']}, {band['high']}), pass {passed}")
    return 0 if passed else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--official":
        return official(int(argv[1]), argv[3:] if len(argv) > 2 and argv[2] == "--" else argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("data", "run", "collect"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.data:
        return data(output)
    if args.run:
        return run(args.device, args.seed, output)
    return collect(output, (args.seed,))


if __name__ == "__main__":
    raise SystemExit(main())
