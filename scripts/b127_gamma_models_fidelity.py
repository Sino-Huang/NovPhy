"""#134 T3 γ-models G-fid: the paper's pendulum value-estimation result (Janner et al., NeurIPS 2020, Fig. 4) through
the official Colab notebook (jannerm/gamma-models @ bb23e06, ``scripts/gamma-pendulum.ipynb``, vendored).

Statistic (Decision P-fid on #134; frozen in ``fidelity/band.json`` before any run): the Spearman rank correlation
between a γ-model value map and the paper's "Ground truth" value map (value iteration on a discretised pendulum),
both on the ``n x n`` state grid of the official ``visualize_values`` (``get_states(n)``: angle linspace(0, 2π, n),
angular velocity linspace(-8, 8, n) reversed), n = 41 (the function's default). The paper's number is the same
correlation for the paper's own "Value estimates" panel, digitised from the arXiv PDF (``--digitise``):

- arXiv 2010.14496v4 (``data/external/gamma_models/paper/2010.14496v4.pdf``), page 9, Fig. 4. ``pdfimages`` gives the
  three tall raster panels: Rewards (343 x 600, PDF object 701), Ground truth (345 x 603, object 692), Value estimates
  (345 x 603, object 693). They are drawn without flips (checked against a 600-dpi render of the page); each spans
  the axes [0, 2π] x [-8, 8] with +8 at the top.
- Colour map: matplotlib ``magma``; every pixel maps to the index of its nearest LUT colour (Euclidean RGB); the
  residual is recorded (max 1.5 / 255). Check: the digitised Rewards panel against the analytic ``reward_fn`` on the
  grid (Spearman 0.9999).
- Grid sampling: the pixel containing (θ, θdot): row = floor((8 - θdot) / 16 · H), col = floor(θ / 2π · W), clipped.

Run (``--train`` then ``--evaluate``; env ``b127-gamma_models``: torch 2.13 + rlkit @ 5d355ba + gym 0.17.3 for the
released SAC policy snapshot; CPU float32, 4 threads): the notebook's cells in order with its ``Args`` (hidden
[256, 256, 256], σ 0.1, batch 1024, Adam 1e-4 / weight decay 1e-5, τ 0.005, discount 0.99, sample discount 0.9,
burn-in discount 0.5 for 2000 steps, 50 000 steps). Deviations (all logged in ``gfid.json``): device ``cpu`` instead of
``cuda:0``; seeds fixed (torch 0, numpy 0; the notebook sets none); the in-loop visualisation / video cell body (every
100 steps; display only, reads the model without touching parameters or RNG) is skipped, the pre-training
``visualize(prob_fn)`` cell runs verbatim (it data-initialises the model's ActNorm layers); ``save_model`` every 1000
steps is replaced by a resumable checkpoint (model, target, optimizer, ActNorm init flags, numpy / torch RNG
states, step) every 1000 steps and on SIGTERM / SIGUSR1 (exit 75); the target network shares the online model's
prior after the deepcopy (torch 2.13 deepcopy shim, VENDOR.json). Readout: the final model, the value map by the
official ``visualize_values`` arithmetic (``V[c] = Σ_q prob_fn(c, q) · reward_fn(q)``) at n = 41 (primary) and
n = 20 (the notebook's last cell; descriptive), plus the official figure itself.

Modes: ``--digitise`` (novphy or gamma env; writes ``fidelity/digitised/``), ``--write-band`` (once; refuses to
overwrite), ``--train`` (resumable), ``--evaluate`` (writes ``fidelity/gfid.json``), ``--run`` (train + evaluate),
``--project`` (CPU s/step micro-benchmark on synthetic states -> ``fidelity/projection.json``).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / ".local-artifacts" / "issue-134-gamma-models-v1"
FID = OUTPUT / "fidelity"
RUN = FID / "run"
DATA = ROOT / "data" / "external" / "gamma_models"
POOL = DATA / "pools" / "pendulum.pkl"
POLICY = DATA / "policies" / "pendulum.pkl"
PAPER = DATA / "paper" / "2010.14496v4.pdf"
PAPER_URL = "https://arxiv.org/pdf/2010.14496v4"
NOTEBOOK = ROOT / "world_model/baselines/gamma_models/vendor/scripts/gamma-pendulum.ipynb"
COMMIT = "bb23e06753717a9255065355b5f6ab77278f3305"
SCHEMA_BAND = "issue_127_gfid_band_v1"
SCHEMA_GFID = "issue_127_gfid_v1"
SCHEMA_DIGITISED = "issue_134_fig4_digitised_v1"
EXIT_PAUSED = 75
SEED = 0
THREADS = 4
GRID_N = 41  # visualize_values' default n_steps (primary)
GRID_N_NOTEBOOK = 20  # the notebook's last cell (descriptive)
DIGITISE_GRIDS = (20, 41, 101)
CHECKPOINT_EVERY = 1000
LOG_EVERY = 100
PANELS = {"rewards": 701, "ground_truth": 692, "value_estimates": 693}  # PDF object ids of the Fig. 4 panels


class Args:
    """The Colab notebook's ``Args`` cell (scripts/gamma-pendulum.ipynb), device aside."""
    data_path = str(POOL)
    policy_path = str(POLICY)
    load_epoch = None
    device = "cpu"  # notebook: 'cuda:0'
    hidden_dims = [256, 256, 256]
    sigma = 0.1
    batch_size = 1024
    lr = 1e-4
    decay = 1e-5
    tau = 0.005
    discount = 0.99
    sample_discount = 0.9
    burnin_discount = 0.5
    n_burnin = 2000
    n_steps = 50000
    vis_freq = 100
    save_freq = 1000


ARGS = {k: v for k, v in vars(Args).items() if not k.startswith("_")}
STOP = []


def log(message):
    print(f"[issue-134 fidelity] {message}", flush=True)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


# ---------------------------------------------------------------------------------------------------------------
# --digitise: the paper's Fig. 4 panels -> value maps on the visualize_values grid
# ---------------------------------------------------------------------------------------------------------------

def magma_index(rgb):
    """[H, W, 3] uint8 -> (index in [0, 255] of the nearest magma LUT colour, residual in RGB/255 units)."""
    from matplotlib import colormaps
    lut = colormaps["magma"](np.linspace(0, 1, 256))[:, :3]
    x = rgb.astype(np.float64) / 255.0
    d = ((x[..., None, :] - lut[None, None]) ** 2).sum(-1)
    return d.argmin(-1), np.sqrt(d.min(-1)) * 255.0


def grid_coordinates(n):
    """The visualize_values state grid (gamma.visualization.pendulum.get_states): rows = angular velocity from +8 to
    -8, columns = angle from 0 to 2π."""
    return np.linspace(0, 2 * np.pi, n), np.linspace(-8, 8, n)[::-1]


def sample_panel(panel, n):
    """Values of a digitised panel [H, W] (top = +8, left = 0) at the n x n grid: the pixel containing each point."""
    h, w = panel.shape
    thetas, thetadots = grid_coordinates(n)
    rows = np.clip(np.floor((8 - thetadots) / 16 * h).astype(int), 0, h - 1)
    cols = np.clip(np.floor(thetas / (2 * np.pi) * w).astype(int), 0, w - 1)
    return panel[rows][:, cols]


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel())[0])


def analytic_rewards(n):
    """reward_fn of the official pendulum visualisation on the grid (same arithmetic)."""
    thetas, thetadots = grid_coordinates(n)
    theta, thetadot = np.meshgrid(thetas, thetadots)
    normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
    return -(normalized ** 2 + .1 * thetadot ** 2)


def extract_panels(pdf, out_dir):
    """pdfimages (poppler) on page 9; the three panels picked by PDF object id (pdfimages -list)."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(["pdfimages", "-list", "-f", "9", "-l", "9", str(pdf)], capture_output=True, text=True,
                             check=True).stdout.splitlines()[2:]
    images = [line.split() for line in listing if line.split()[2] == "image"]
    first = int(listing[0].split()[1])
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdfimages", "-png", "-f", "9", "-l", "9", str(pdf), f"{tmp}/p"], check=True)
        panels = {}
        for name, obj in PANELS.items():
            row = [r for r in images if int(r[10]) == obj]
            if len(row) != 1:
                raise ValueError(f"PDF object {obj} not found on page 9")
            num = int(row[0][1]) - first
            image = Image.open(f"{tmp}/p-{num:03d}.png").convert("RGB")
            image.save(out_dir / f"fig4-{name}.png")
            panels[name] = {"object": obj, "pdfimages_num": int(row[0][1]), "size": list(image.size),
                            "rgb": np.asarray(image)}
    return panels


def digitise(pdf=PAPER, out_dir=FID / "digitised"):
    out_dir = Path(out_dir)
    panels = extract_panels(pdf, out_dir)
    maps, residual = {}, {}
    for name, panel in panels.items():
        index, res = magma_index(panel["rgb"])
        maps[name] = index.astype(np.float64) / 255.0
        residual[name] = {"mean": float(res.mean()), "max": float(res.max())}
    record = {"schema": SCHEMA_DIGITISED, "pdf": rel(pdf), "pdf_sha256": sha256_of(pdf), "pdf_url": PAPER_URL,
              "page": 9, "figure": 4, "colormap": "magma (nearest of 256 LUT colours)",
              "panels": {k: {kk: vv for kk, vv in v.items() if kk != "rgb"} | {"magma_residual_rgb255": residual[k],
                                                                              "png": rel(out_dir / f"fig4-{k}.png")}
                         for k, v in panels.items()},
              "orientation": "identity: each raster spans angle [0, 2π] (left to right) x angular velocity [+8, -8] "
                             "(top to bottom); checked against a 600-dpi render of page 9 (mean abs RGB difference "
                             "0.6 / 1.4 / 0.8 for ground truth / value estimates / rewards vs 26.8+ for any flip)",
              "sampling": "pixel containing (θ, θdot): row = floor((8 - θdot) / 16 H), col = floor(θ / 2π W), clipped",
              "grids": {}}
    for n in DIGITISE_GRIDS:
        gt = sample_panel(maps["ground_truth"], n)
        record["grids"][str(n)] = {
            "rho_value_estimates_vs_ground_truth": spearman(sample_panel(maps["value_estimates"], n), gt),
            "rho_analytic_reward_vs_ground_truth": spearman(analytic_rewards(n), gt),
            "rho_reward_panel_vs_analytic_reward": spearman(sample_panel(maps["rewards"], n), analytic_rewards(n))}
    np.savez_compressed(out_dir / "fig4-maps.npz", **maps)
    record["maps_npz"] = rel(out_dir / "fig4-maps.npz")
    record["maps_npz_sha256"] = sha256_of(out_dir / "fig4-maps.npz")
    record["written_at"] = utc_now()
    write_json(out_dir / "fig4.json", record)
    log(json.dumps(record["grids"]))
    return record


def ground_truth_grid(n, maps_npz=None):
    maps = np.load(maps_npz or (FID / "digitised" / "fig4-maps.npz"))
    return sample_panel(maps["ground_truth"], n), sample_panel(maps["value_estimates"], n)


MARGIN = 0.15


def write_band():
    path = FID / "band.json"
    if path.is_file():
        raise SystemExit(f"{rel(path)} exists (frozen); never rewritten")
    record = read_json(FID / "digitised" / "fig4.json")
    paper = record["grids"][str(GRID_N)]["rho_value_estimates_vs_ground_truth"]
    myopic = record["grids"][str(GRID_N)]["rho_analytic_reward_vs_ground_truth"]
    low = round(paper - MARGIN, 3)
    band = {
        "schema": SCHEMA_BAND, "issue": 134, "arm_family": "gamma_models", "tier": "E", "frozen_at": utc_now()[:10],
        "paper": {"source": "Janner, Mordatch, Levine, γ-Models: Generative Temporal Difference Learning for "
                            "Infinite-Horizon Prediction, NeurIPS 2020 (arXiv 2010.14496v4), Fig. 4 (pendulum, "
                            "γ = 0.99: 'Value estimates' vs 'Ground truth' = value iteration on a discretisation)",
                  "value": paper, "digitised": rel(FID / "digitised" / "fig4.json"),
                  "digitised_sha256": sha256_of(FID / "digitised" / "fig4.json"),
                  "resolution_sensitivity": {n: record["grids"][n]["rho_value_estimates_vs_ground_truth"]
                                             for n in record["grids"]},
                  "myopic_reference": {"rho_reward_vs_ground_truth": myopic,
                                       "meaning": "a value map equal to the one-step reward (no long-horizon "
                                                  "prediction)"}},
        "statistic": f"Spearman rank correlation, on the visualize_values grid n = {GRID_N} (get_states: angle "
                     "linspace(0, 2π, n), angular velocity linspace(-8, 8, n) reversed), between the run's value map "
                     "V[c] = Σ_q prob_fn(c, q) · reward_fn(q) (the official visualize_values arithmetic, final "
                     "model) and the digitised Ground-truth panel sampled at the same states",
        "band": [low, 1.0],
        "band_rule": f"[paper - {MARGIN}, 1]: the paper's single unseeded run gives no spread; the margin covers the "
                     f"digitisation / resolution sensitivity of the paper's own map (rho {min(record['grids'][n]['rho_value_estimates_vs_ground_truth'] for n in record['grids']):.3f}-"
                     f"{max(record['grids'][n]['rho_value_estimates_vs_ground_truth'] for n in record['grids']):.3f} over grids 20 / 41 / 101) "
                     "and run-to-run variation of a bootstrapped γ = 0.99 model; the lower bound stays far above the "
                     f"myopic reference ({myopic:.3f}), so a model without long-horizon prediction cannot pass. "
                     "Upper bound 1: agreeing with the ground truth more closely than the paper's figure is not a "
                     "reproduction failure.",
        "pass_rule": f"pass iff the run's rho at n = {GRID_N} lies in the band (inclusive). One run (seed 0). At most "
                     "two logged repair attempts, each fixing a diagnosed port / harness defect, never the band "
                     "(P-fid); then T3 is dropped with fidelity_not_reproduced.",
        "protocol": {"notebook": rel(NOTEBOOK), "notebook_sha256": sha256_of(NOTEBOOK), "commit": COMMIT,
                     "args": ARGS, "seeds": {"torch": SEED, "numpy": SEED}, "device": "cpu float32",
                     "threads": THREADS, "runs": 1,
                     "deviations": DEVIATIONS},
        "posted_before_run": True,
    }
    write_json(path, band)
    log(f"band.json frozen: {band['band']} ({sha256_of(path)})")
    return band


DEVIATIONS = [
    "device cpu (notebook: cuda:0); torch 2.13.0 float32, 4 threads",
    "seeds fixed: torch.manual_seed(0), np.random.seed(0) before the model cell (the notebook sets none)",
    "in-loop visualisation cell body (every 100 steps: visualize + display_video) skipped: display only; it reads "
    "the model without touching parameters, optimizer or RNG; the loss is logged instead",
    "save_model every 1000 steps replaced by a resumable checkpoint (model, target, optimizer, ActNorm init flags, "
    "numpy / torch RNG states, step) every 1000 steps and on SIGTERM / SIGUSR1 (exit 75)",
    "target network shares the online model's prior after copy.deepcopy (torch 2.13 deepcopy shim; VENDOR.json)",
    "Policy snapshot loaded with map_location=cpu, weights_only=False (vendored P-route edit; VENDOR.json)",
]


# ---------------------------------------------------------------------------------------------------------------
# --train (the notebook, resumable) and --evaluate
# ---------------------------------------------------------------------------------------------------------------

def _official():
    from world_model.baselines.gamma_models.vendor.gamma.flows import make_conditional_flow
    from world_model.baselines.gamma_models.vendor.gamma.td.distributions import BootstrapTarget
    from world_model.baselines.gamma_models.vendor.gamma.td.structs import ReplayPool, Policy
    from world_model.baselines.gamma_models.vendor.gamma.td.utils import soft_update_from_to, format_batch
    from world_model.baselines.gamma_models.vendor.gamma.utils import mkdir, set_device
    from world_model.baselines.gamma_models.vendor.gamma.visualization import make_prob_fn
    from world_model.baselines.gamma_models.vendor.gamma.visualization import pendulum
    return dict(make_conditional_flow=make_conditional_flow, BootstrapTarget=BootstrapTarget, ReplayPool=ReplayPool,
                Policy=Policy, soft_update_from_to=soft_update_from_to, format_batch=format_batch, mkdir=mkdir,
                set_device=set_device, make_prob_fn=make_prob_fn, pendulum=pendulum)


def actnorm_flags(model):
    return [bool(getattr(f, "data_dep_init_done", False)) for f in model.flow.flows]


def set_actnorm_flags(model, flags):
    for flow, flag in zip(model.flow.flows, flags):
        if hasattr(flow, "data_dep_init_done"):
            flow.data_dep_init_done = flag


def _request_stop(signum, frame):
    STOP.append(signum)


def build(o, args=Args):
    """The notebook's data and model cells (seeded)."""
    o["set_device"](args.device)
    dataset = o["ReplayPool"](args.data_path)
    policy = o["Policy"](args.policy_path)
    observation_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]
    condition_dims = {"s": observation_dim, "a": action_dim}
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = o["make_conditional_flow"](observation_dim, args.hidden_dims, condition_dims)
    target_model = copy.deepcopy(model)
    target_model.prior = model.prior  # torch 2.13 deepcopy shim (VENDOR.json runtime note)
    bootstrap = o["BootstrapTarget"](target_model, args.discount)
    return dataset, policy, model, target_model, bootstrap


def save_checkpoint(path, step, model, target_model, optimizer, elapsed):
    payload = {"step": step, "model": model.state_dict(), "target": target_model.state_dict(),
               "optimizer": optimizer.state_dict(), "model_actnorm": actnorm_flags(model),
               "target_actnorm": actnorm_flags(target_model), "torch_rng": torch.get_rng_state(),
               "numpy_rng": np.random.get_state(), "elapsed": elapsed, "saved_at": utc_now()}
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(run_dir=RUN, args=Args, max_steps=None):
    """The notebook's training cells (Colab version), resumable; returns 0 when finished, 75 when paused."""
    torch.set_num_threads(THREADS)
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, _request_stop)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "model-final.pt"
    if final.is_file():
        log("model-final.pt exists; nothing to train")
        return 0
    o = _official()
    dataset, policy, model, target_model, bootstrap = build(o, args)
    prob_fn = o["make_prob_fn"](model, policy)
    checkpoint = run_dir / "checkpoint.pt"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = torch.nn.MSELoss()
    start, elapsed = 0, 0.0
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        target_model.load_state_dict(state["target"])
        optimizer.load_state_dict(state["optimizer"])
        set_actnorm_flags(model, state["model_actnorm"])
        set_actnorm_flags(target_model, state["target_actnorm"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        start, elapsed = state["step"], state["elapsed"]
        log(f"resumed at step {start}")
    else:
        ## visualize the randomly-initialized model (notebook cell; data-initialises the model's ActNorm)
        o["pendulum"].visualize(prob_fn, save_path=str(run_dir / "init"))
    stop_at = args.n_steps if max_steps is None else min(args.n_steps, start + max_steps)
    began = time.monotonic()
    i = start - 1
    with open(run_dir / "train-log.jsonl", "a") as log_file:
        for i in range(start, stop_at):
            # ---- notebook training cell body (verbatim apart from the logging / checkpoint lines) ----
            if i < args.n_burnin:
                bootstrap.update_discount(args.burnin_discount)
                sample_discount = args.burnin_discount
            else:
                bootstrap.update_discount(args.discount)
                sample_discount = args.sample_discount
            batch = dataset.sample(args.batch_size)
            condition_dict, next_condition_dict = o["format_batch"](batch, policy)
            bootstrap.update_p(next_condition_dict["s"], sigma=args.sigma)
            samples = bootstrap.sample(args.batch_size, condition_dict, next_condition_dict,
                                       discount=sample_discount)
            log_prob_target = bootstrap.log_prob(samples, condition_dict, next_condition_dict)
            log_prob_model = model.log_prob(samples, condition_dict)
            loss = criterion(log_prob_model, log_prob_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            o["soft_update_from_to"](model, target_model, args.tau)
            # ---- end of the notebook body ----
            if i % LOG_EVERY == 0 or i == args.n_steps - 1:
                log_file.write(json.dumps({"step": i, "loss": float(loss.item()),
                                           "seconds": elapsed + time.monotonic() - began}) + "\n")
                log_file.flush()
            stop = bool(STOP) and i + 1 < args.n_steps
            if stop or ((i + 1) % CHECKPOINT_EVERY == 0 and i + 1 < args.n_steps):
                save_checkpoint(checkpoint, i + 1, model, target_model, optimizer,
                                elapsed + time.monotonic() - began)
            if stop:
                log(f"stop signal: checkpoint at step {i + 1}; exit {EXIT_PAUSED}")
                return EXIT_PAUSED
    seconds = elapsed + time.monotonic() - began
    if i + 1 < args.n_steps:
        save_checkpoint(checkpoint, i + 1, model, target_model, optimizer, seconds)
        return 0
    payload = {"model": model.state_dict(), "model_actnorm": actnorm_flags(model), "steps": args.n_steps,
               "final_loss": float(loss.item()), "seconds": seconds, "args": ARGS, "seed": SEED,
               "torch": torch.__version__, "threads": THREADS, "finished_at": utc_now()}
    tmp = Path(str(final) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, final)
    if checkpoint.is_file():
        checkpoint.unlink()
    log(f"finished {args.n_steps} steps in {seconds:.0f}s (final loss {payload['final_loss']:.6f})")
    return 0


@torch.no_grad()
def value_map(model, policy, n, o):
    """visualize_values' arithmetic (gamma/visualization/pendulum.py) without the plot: V [n, n]."""
    queries, *_ = o["pendulum"].get_states(n)
    rewards = o["pendulum"].reward_fn(queries)
    prob_fn = o["make_prob_fn"](model, policy)
    values = []
    for cond in queries:
        cond_rep = cond[None].repeat(len(queries), axis=0)
        probs = prob_fn(cond_rep, queries)
        values.append((probs * rewards).sum(-1))
    return np.array(values).reshape(n, n)


def evaluate(run_dir=RUN):
    torch.set_num_threads(THREADS)
    band_path = FID / "band.json"
    band = read_json(band_path)
    run_dir = Path(run_dir)
    final = run_dir / "model-final.pt"
    if not final.is_file():
        raise SystemExit("model-final.pt missing; run --train first")
    o = _official()
    o["set_device"]("cpu")
    payload = torch.load(final, map_location="cpu", weights_only=False)
    dataset_obs_dim, condition_dims = 3, {"s": 3, "a": 1}
    model = o["make_conditional_flow"](dataset_obs_dim, Args.hidden_dims, condition_dims)
    model.load_state_dict(payload["model"])
    set_actnorm_flags(model, payload["model_actnorm"])
    model.eval()
    policy = o["Policy"](Args.policy_path)
    result = {}
    for n in (GRID_N, GRID_N_NOTEBOOK):
        began = time.monotonic()
        values = value_map(model, policy, n, o)
        gt, paper_values = ground_truth_grid(n)
        np.save(run_dir / f"value-map-{n}.npy", values)
        result[str(n)] = {"rho_vs_ground_truth": spearman(values, gt),
                          "rho_vs_paper_value_estimates": spearman(values, paper_values),
                          "finite": bool(np.isfinite(values).all()), "seconds": time.monotonic() - began,
                          "value_map": rel(run_dir / f"value-map-{n}.npy")}
    o["pendulum"].visualize_values(o["make_prob_fn"](model, policy), n_steps=GRID_N_NOTEBOOK,
                                   save_path=str(run_dir / "values-official-20"))
    rho = result[str(GRID_N)]["rho_vs_ground_truth"]
    low, high = band["band"]
    passed = bool(np.isfinite(rho) and low <= rho <= high)
    log_rows = [json.loads(line) for line in (run_dir / "train-log.jsonl").read_text().splitlines()]
    gfid = {"schema": SCHEMA_GFID, "issue": 134, "arm": "T3 γ-model-car", "tier": "E",
            "environment": "pendulum (released replay pool + SAC policy), official Colab notebook",
            "commit": COMMIT, "band": band["band"], "band_sha256": sha256_of(band_path),
            "paper_value": band["paper"]["value"], "statistic": band["statistic"],
            "runs": [{"seed": SEED, "rho": rho, "grids": result, "model": rel(final), "model_sha256": sha256_of(final),
                      "steps": payload["steps"], "final_loss": payload["final_loss"],
                      "wall_seconds": payload["seconds"], "device": "cpu", "threads": THREADS,
                      "loss_trace_tail": log_rows[-5:]}],
            "compute": {"cpu_seconds_wall": payload["seconds"], "gpu_seconds": 0.0},
            "deviations": DEVIATIONS, "torch": torch.__version__, "pass": passed, "G_fid": passed,
            "decision": "https://github.com/Sino-Huang/NovPhy/issues/134#issuecomment-6390881384",
            "script_sha256": sha256_of(Path(__file__)), "digitised_sha256": sha256_of(FID / "digitised" / "fig4.json"),
            "attempts": [{"log": rel(p), "log_sha256": sha256_of(p)} for p in sorted(run_dir.glob("attempt*.log"))],
            "repairs": [], "complete": True, "written_at": utc_now()}
    write_json(FID / "gfid.json", gfid)
    log(f"G-fid: rho {rho:.4f} vs band {band['band']} -> pass {passed}")
    return 0 if passed else 1


# ---------------------------------------------------------------------------------------------------------------
# --project: CPU micro-benchmark (synthetic states; not a fidelity run)
# ---------------------------------------------------------------------------------------------------------------

def project(steps=20, gpu_over_cpu=(0.05, 0.36)):
    """CPU s/step of the notebook update on synthetic pendulum-shaped batches (random states; a fixed linear
    stand-in policy), scaled to GPU-h by an [INFERENCE] GPU/CPU ratio range (lower: small-MLP flows gain ~20x on a
    GPU; upper: #133's measured THICK ratio 0.36)."""
    torch.set_num_threads(THREADS)
    o = _official()
    o["set_device"]("cpu")
    torch.manual_seed(SEED)
    model = o["make_conditional_flow"](3, Args.hidden_dims, {"s": 3, "a": 1})
    target_model = copy.deepcopy(model)
    target_model.prior = model.prior
    bootstrap = o["BootstrapTarget"](target_model, Args.discount)
    optimizer = torch.optim.Adam(model.parameters(), lr=Args.lr, weight_decay=Args.decay)
    criterion = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(1)
    times = []
    for i in range(steps):
        s = torch.rand(Args.batch_size, 3, generator=g) * 2 - 1
        a = torch.tanh(s[:, :1])
        s2 = s + 0.05 * torch.randn(Args.batch_size, 3, generator=g)
        c, c2 = {"s": s, "a": a}, {"s": s2, "a": torch.tanh(s2[:, :1])}
        t = time.monotonic()
        bootstrap.update_discount(Args.discount)
        bootstrap.update_p(c2["s"], sigma=Args.sigma)
        samples = bootstrap.sample(Args.batch_size, c, c2, discount=Args.sample_discount)
        loss = criterion(model.log_prob(samples, c), bootstrap.log_prob(samples, c, c2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        o["soft_update_from_to"](model, target_model, Args.tau)
        times.append(time.monotonic() - t)
    cpu = float(np.median(times[2:]))
    cpu_h = cpu * Args.n_steps / 3600
    record = {"schema": "issue_134_fidelity_projection_v1", "cpu_seconds_per_step": cpu, "threads": THREADS,
              "steps": Args.n_steps, "cpu_hours_training": cpu_h,
              "evaluation_cpu_hours_estimate": 0.25,
              "gpu_hours_training": [cpu_h * gpu_over_cpu[0], cpu_h * gpu_over_cpu[1]],
              "label": "[INFERENCE] CPU micro-benchmark (synthetic pendulum-shaped batches, 4 threads); the GPU range "
                       "applies an assumed GPU/CPU per-step ratio of 0.05-0.36 (upper = #133's measured THICK ratio)",
              "cpu_feasible": bool(cpu_h <= 12.0), "measured_at": utc_now()}
    write_json(FID / "projection.json", record)
    log(f"projection: {cpu:.3f} s/step CPU -> {cpu_h:.2f} CPU-h training; GPU {record['gpu_hours_training']}")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("digitise", "write-band", "train", "evaluate", "run", "project"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="--train: bound this call (tests)")
    args = parser.parse_args(argv)
    if args.digitise:
        digitise()
        return 0
    if args.write_band:
        write_band()
        return 0
    if args.project:
        project()
        return 0
    if args.train or args.run:
        if not (FID / "band.json").is_file():
            raise SystemExit("band.json missing: the band is frozen and posted (Decision P-fid) before any run")
        code = train(max_steps=args.max_steps)
        if code != 0 or args.train:
            return code
    return evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
