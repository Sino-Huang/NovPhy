"""#131 D1 THICK G-fid: the frozen band from the paper, the GPU feasibility probes and the costed plan.

Modes (no heavy import at module level):

``--band`` (any env with PyMuPDF, e.g. ``uv run --no-project --with pymupdf --with numpy python ...``)
    Reads Fig. 8 of the THICK paper (ICLR 2024, proceedings PDF pinned by sha256 in ``data/external/thick``) from its
    vector paths: per VisualPinPad task (Three, Four, Five) and method (THICK Dreamer, Dreamer = DreamerV2), the
    seed-mean evaluation-return curve (one point per 1e4 env steps) and its +-1 standard-error band (7 seeds). Axis
    calibration from the figure's own grid lines / tick labels. Writes ``fidelity/band.json`` (schema
    ``issue_127_gfid_band_v1``) with the final visible step's mean, SE and SD = SE * sqrt(7), the steps to the
    paper's 95 % thresholds (Table 3: 359 PinPadFour, 274 PinPadFive), the frozen band and the pass rule.

``--tf-gpu-probe OUT_JSON [--official-timing]`` (a TF env; scheduler job ``131-tf-gpu-probe``)
    Can this TensorFlow build run on the RTX 5090 (sm_120)? Lists devices, runs a matmul / conv / GRU-sized step,
    and with ``--official-timing`` times the official THICK world-model + high-level training step
    (``thick_pinpad`` config, batch 16 x 50, ``jit`` as upstream) on the GPU.

``--port-timing OUT_JSON`` (env ``novphy``, scheduler job)
    Times the PyTorch port's world-model + high-level training step on the GPU (VisualPinPad image config, batch
    16 x 50; NovPhy carrier config, batch 16 x 46), float32.

``--plan`` (env ``novphy``)
    Writes ``fidelity/plan.json``: the minimal G-fid design and its GPU-h projection from the measured timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-131-thick-v1"
FIDELITY = OUTPUT / "fidelity"
PAPER = ROOT / "data/external/thick/paper/thick_iclr2024.pdf"
PAPER_URL = ("https://proceedings.iclr.cc/paper_files/paper/2024/file/"
             "13b45b44e26c353c64cba9529bf4724f-Paper-Conference.pdf")
SEEDS_PAPER = 7
COLORS = {(0.149, 0.439, 0.51): "dreamer", (0.608, 0.212, 0.243): "thick", (0.518, 0.467, 0.443): "director"}
# Fig. 8 panels (page 8): x data range from the tick labels, y from the grid lines (PDF points)
PANELS = {
    "three": {"x_px": (127.5, 230.37), "x_val": (0.0, 4e5), "y_px": (640.3, 602.81), "y_val": (0.0, 600.0),
              "window_end": 4e5},
    "four": {"x_px": (260.95, 363.82), "x_val": (0.0, 4e5), "y_px": (639.82, 601.01), "y_val": (0.0, 400.0),
             "window_end": 4e5},
    "five": {"x_px": (392.16, 495.03), "x_val": (1e5, 6e5), "y_px": (639.82, 602.0), "y_val": (0.0, 300.0),
             "window_end": 6e5},
}
THRESHOLDS = {"four": 359.0, "five": 274.0}  # paper Suppl. F.5 / Table 3: 95 % of the highest mean reward
TABLE3_1M = {"four": {"thick": 140e3, "dreamer": 200e3}, "five": {"thick": 340e3, "dreamer": 590e3}}


def sha256_of(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# band (paper Fig. 8)
# ---------------------------------------------------------------------------


def _polyline(drawing):
    points = []
    for item in drawing["items"]:
        if item[0] != "l":
            raise ValueError(f"unexpected path item {item[0]}")
        start, end = (item[1].x, item[1].y), (item[2].x, item[2].y)
        if not points or points[-1] != start:
            points.append(start)
        points.append(end)
    return points


def digitize_fig8():
    import numpy as np
    import pymupdf

    page = pymupdf.open(PAPER)[7]
    if "Figure 8: VisualPinPad results" not in page.get_text():
        raise ValueError("page 8 does not carry Fig. 8")
    raw = {}
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if not 595 < rect.y0 < 650:
            continue
        color = drawing.get("color") if drawing["type"] == "s" else drawing.get("fill")
        if color is None:
            continue
        method = COLORS.get(tuple(round(c, 3) for c in color))
        if method is None or len(drawing["items"]) < 10:
            continue
        task = min(PANELS, key=lambda t: abs(rect.x0 - PANELS[t]["x_px"][0]) if rect.x0 < PANELS[t]["x_px"][1] + 5
                   else math.inf)
        raw.setdefault(task, {}).setdefault(method, {})["line" if drawing["type"] == "s" else "fill"] = \
            _polyline(drawing)
    curves = {}
    for task, cal in PANELS.items():
        (x0, x1), (v0, v1) = cal["x_px"], cal["x_val"]
        (y0, y1), (r0, r1) = cal["y_px"], cal["y_val"]
        to_x = lambda px: v0 + (px - x0) / (x1 - x0) * (v1 - v0)
        to_y = lambda py: r0 + (y0 - py) / (y0 - y1) * (r1 - r0)
        for method in ("thick", "dreamer"):
            line = np.asarray(raw[task][method]["line"])
            fill = np.asarray(raw[task][method]["fill"])
            n = len(line)
            # the SE band is one closed polygon: a duplicated first point, the upper bound forward, the lower back
            upper, lower = fill[1:n + 1], fill[n + 1:2 * n + 1][::-1]
            if not (np.allclose(upper[:, 0], line[:, 0], atol=1e-3) and np.allclose(lower[:, 0], line[:, 0], atol=1e-3)):
                raise ValueError(f"{task}/{method}: SE polygon does not follow the mean curve")
            steps = np.round(to_x(line[:, 0]) / 1e4) * 1e4
            mean = to_y(line[:, 1])
            se = np.abs(to_y(upper[:, 1]) - to_y(lower[:, 1])) / 2
            curves[f"{task}/{method}"] = {"steps": steps.tolist(), "mean": np.round(mean, 2).tolist(),
                                          "se": np.round(se, 2).tolist()}
    return curves


def first_crossing(steps, mean, threshold):
    for s, m in zip(steps, mean):
        if m >= threshold:
            return s
    return None


def band():
    import numpy as np

    curves = digitize_fig8()
    tasks = {}
    for task, cal in PANELS.items():
        entry = {"window_end_steps": cal["window_end"]}
        for method in ("thick", "dreamer"):
            c = curves[f"{task}/{method}"]
            steps, mean, se = np.asarray(c["steps"]), np.asarray(c["mean"]), np.asarray(c["se"])
            i = int(np.nonzero(steps <= cal["window_end"])[0][-1])
            sd = float(se[i]) * math.sqrt(SEEDS_PAPER)
            visible = steps <= cal["window_end"]
            entry[method] = {"final_step": float(steps[i]), "mean": round(float(mean[i]), 1),
                             "se": round(float(se[i]), 1), "sd": round(sd, 1),
                             "band": [round(float(mean[i]) - 2 * sd, 1), round(float(mean[i]) + 2 * sd, 1)],
                             "window_mean_return": round(float(mean[visible].mean()), 1),
                             "points_beyond_window": int((~visible).sum())}
            if task in THRESHOLDS:
                crossing = first_crossing(steps, mean, THRESHOLDS[task])
                entry[method]["steps_to_threshold"] = crossing
                entry[method]["table3_1M"] = TABLE3_1M[task][method]
        if task in THRESHOLDS:
            entry["threshold"] = THRESHOLDS[task]
            entry["advantage_steps"] = entry["dreamer"]["steps_to_threshold"] - entry["thick"]["steps_to_threshold"]
        tasks[task] = entry
    payload = {
        "schema": "issue_127_gfid_band_v1", "issue": 131, "arm": "D1 THICK",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"paper": "Gumbsch, Sajid, Martius, Butz. Learning Hierarchical World Models with Adaptive Temporal "
                            "Abstractions from Discrete Latent Dynamics. ICLR 2024 (OpenReview TjCDNssXKU).",
                   "pdf_url": PAPER_URL, "pdf": str(PAPER.relative_to(ROOT)), "pdf_sha256": sha256_of(PAPER),
                   "figure": "Fig. 8 (p. 8): VisualPinPad mean evaluation returns, 7 seeds, +- standard error; "
                             "replay buffers pre-filled with 1M Plan2Explore steps (Suppl. F.4)",
                   "table": "Table 3 (Suppl. F.5): env steps to 95 % of the max mean reward (1M exploration: "
                            "PinPadFour THICK 140k / Dreamer 200k; PinPadFive 340k / 590k)",
                   "arxiv": "none: arXiv 2310.05167 is Hieros (Mattes et al.), not THICK; no arXiv version found"},
        "reading": ("vector paths of the PDF (PyMuPDF get_drawings): the stroked polyline of each method is the seed "
                    "mean, the filled polygon of the same colour is mean +- 1 SE; x from the tick labels "
                    "(1e4-step grid, the points are every 1e4 env steps), y from the grid lines; SD = SE * sqrt(7). "
                    "Check: the digitised seed-mean curves cross the Table-3 thresholds exactly at the Table-3 "
                    "steps (1M exploration column). The PDF keeps points beyond the plotted x window "
                    "(clipped); the band uses the last point inside the window, as a reader of the figure would."),
        "curves": curves, "tasks": tasks,
        "pass_rule": {
            "task": "the G-fid task(s) named in fidelity/plan.json (frozen before the run)",
            "returns": "for THICK Dreamer and for DreamerV2 separately: our seed-mean evaluation return at the "
                       "task's final window step lies inside [paper mean - 2 SD, paper mean + 2 SD]",
            "advantage": "THICK Dreamer's seed-mean evaluation-return curve (one evaluation episode per 1e4 env "
                         "steps, as upstream eval_every / eval_eps) reaches the task's Table-3 threshold strictly "
                         "earlier than DreamerV2's (Dreamer never reaching it counts as later)",
            "pass": "returns (both methods) and advantage",
            "attempts": "at most two logged repair attempts, each fixing a diagnosed port defect, never the band "
                        "(rule P-fid); then D1 is dropped with fidelity_not_reproduced",
        },
    }
    for task, entry in tasks.items():
        if task in TABLE3_1M:
            for method in ("thick", "dreamer"):
                if entry[method]["steps_to_threshold"] != TABLE3_1M[task][method]:
                    raise ValueError(f"{task}/{method}: digitised crossing {entry[method]['steps_to_threshold']} "
                                     f"!= Table 3 {TABLE3_1M[task][method]}")
    write_json(FIDELITY / "band.json", payload)
    for task, entry in tasks.items():
        print(f"[b127-thick-band] {task}: THICK {entry['thick']['mean']} +- {entry['thick']['sd']} (SD) band "
              f"{entry['thick']['band']}; Dreamer {entry['dreamer']['mean']} +- {entry['dreamer']['sd']} band "
              f"{entry['dreamer']['band']}" + (f"; steps to {entry['threshold']}: THICK "
                                                f"{entry['thick']['steps_to_threshold']:.0f} vs Dreamer "
                                                f"{entry['dreamer']['steps_to_threshold']:.0f}"
                                                if "threshold" in entry else ""), flush=True)
    print(f"[b127-thick-band] wrote {FIDELITY / 'band.json'} {sha256_of(FIDELITY / 'band.json')}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# GPU feasibility probes (scheduler jobs)
# ---------------------------------------------------------------------------


def _timed(fn, warmup, repeats, sync):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(repeats):
        fn()
    sync()
    return (time.time() - t0) / repeats


def tf_gpu_probe(out_json, official_timing):
    import numpy as np
    import tensorflow as tf

    result = {"tf": tf.__version__, "python": sys.version.split()[0],
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "build": {k: str(v) for k, v in tf.sysconfig.get_build_info().items()}
              if hasattr(tf.sysconfig, "get_build_info") else None}
    t0 = time.time()
    gpus = tf.config.list_physical_devices("GPU")
    result["gpus"] = [g.name for g in gpus]
    result["details"] = [tf.config.experimental.get_device_details(g) for g in gpus] \
        if hasattr(tf.config.experimental, "get_device_details") else None
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    checks = {}
    try:
        with tf.device("/GPU:0" if gpus else "/CPU:0"):
            a = tf.random.normal([2048, 2048])
            b = tf.random.normal([2048, 2048])
            c = tf.matmul(a, b)
            ref = np.asarray(a) @ np.asarray(b)
            checks["matmul_max_abs_err"] = float(np.abs(np.asarray(c) - ref).max())
            x = tf.random.normal([16, 64, 64, 3])
            k = tf.random.normal([4, 4, 3, 48])
            y = tf.nn.conv2d(x, k, 2, "VALID")
            checks["conv_shape"] = list(y.shape)
            checks["conv_device"] = y.device
            checks["matmul_device"] = c.device
        checks["ok"] = bool(gpus) and "GPU" in checks["matmul_device"]
    except Exception as exc:  # noqa: BLE001 - the probe records any failure
        checks["ok"] = False
        checks["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    checks["first_ops_seconds"] = time.time() - t0
    result["checks"] = checks
    if official_timing and checks.get("ok"):
        try:
            result["official_timing"] = _official_timing()
        except Exception as exc:  # noqa: BLE001
            result["official_timing"] = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
    write_json(out_json, result)
    print(json.dumps({k: v for k, v in result.items() if k != "build"}, indent=1)[:4000], flush=True)
    return 0


def _official_timing():
    """Official THICK world model + high level, thick_pinpad config, batch 16 x 50, one Agent.train-style step."""
    import gym
    import numpy as np
    import tensorflow as tf

    official = ROOT / "world_model/baselines/thick/vendor/thick"
    os.chdir(official)
    sys.path.insert(0, str(official))
    import ruamel.yaml as yaml
    import agent
    import common

    configs = yaml.safe_load((official / "configs.yaml").read_text())
    config = common.Config(configs["defaults"]).update(configs["visualpinpad"]).update(configs["thick_pinpad"])
    obs_space = {"image": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)}
    obs_space.update({k: gym.spaces.Box(0, 1, (), np.float32) for k in ("reward", "is_first", "is_last",
                                                                          "is_terminal")})
    act_space = gym.spaces.Box(0, 1, (5,), np.float32)
    act_space.n = 5
    wm = agent.WorldModel(config, obs_space, act_space, tf.Variable(0, dtype=tf.int64), actor=None)
    rng = np.random.default_rng(0)
    b, t = 16, 50
    data = {"image": tf.constant(rng.integers(0, 256, (b, t, 64, 64, 3), dtype=np.uint8)),
            "action": tf.constant(np.eye(5, dtype=np.float32)[rng.integers(0, 5, (b, t))]),
            "reward": tf.constant(np.zeros((b, t), np.float32)),
            "is_first": tf.constant(np.eye(t, dtype=bool)[np.zeros(b, int)]),
            "is_last": tf.constant(np.zeros((b, t), bool)), "is_terminal": tf.constant(np.zeros((b, t), bool))}

    @tf.function
    def step(batch):
        _, _, metrics = wm.train(batch)
        thick = wm.train_thick(batch)
        return metrics["model_loss"], thick["thick_net_loss"]

    t0 = time.time()
    step(data)
    compile_seconds = time.time() - t0
    sync = lambda: float(wm.rssm.variables[0][0, 0].numpy())
    per_step = _timed(lambda: step(data), 2, 10, sync)
    return {"config": "visualpinpad + thick_pinpad", "batch": [b, t], "first_step_seconds": compile_seconds,
            "seconds_per_wm_and_thick_update": per_step}


def port_timing(out_json, device_name="cuda", repeats=10):
    """GPU seconds per update of the port (world model + high level), per component, TF32 off and on; plus a
    proxy of the THICK-Dreamer behaviour update (15-step policy imagination from 800 starts, actor / critic /
    coarse-critic MLP steps, 16-step high-level imagination) and of one acting step (batch 1)."""
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    from world_model.baselines.thick import novphy as nv
    from world_model.baselines.thick import port

    device = torch.device(device_name)
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    peak = (lambda: torch.cuda.max_memory_allocated() / 2 ** 30) if cuda else (lambda: None)
    result = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if cuda else "cpu",
              "float32": True, "repeats": repeats}
    rng = np.random.default_rng(0)
    gen = torch.Generator(device=device).manual_seed(0)
    noise = port.TorchNoise(gen)
    config = port.load_config(["visualpinpad", "thick_pinpad"])
    obs = {"image": (64, 64, 3), "reward": (), "is_first": (), "is_last": (), "is_terminal": ()}
    b, t = 16, 50
    is_first = torch.zeros(b, t, dtype=torch.bool, device=device)
    is_first[:, 0] = True
    data = {"image": torch.as_tensor(rng.integers(0, 256, (b, t, 64, 64, 3), dtype=np.uint8), device=device),
            "action": torch.as_tensor(np.eye(5, dtype=np.float32)[rng.integers(0, 5, (b, t))], device=device),
            "reward": torch.zeros(b, t, device=device), "is_first": is_first,
            "is_last": torch.zeros(b, t, dtype=torch.bool, device=device),
            "is_terminal": torch.zeros(b, t, dtype=torch.bool, device=device)}
    def measure(tf32):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.manual_seed(0)
        wm = port.WorldModel(config, obs, 5, cont_action=False).to(device)
        model_opt, thick_opt = wm.make_optimizers()
        feat_dim, coarse_dim = wm.rssm.feat_dim, wm.rssm.coarse_dim
        actor = port.MLP(feat_dim, [5], 4, 400, dist="onehot").to(device)
        critic = port.MLP(feat_dim, [], 4, 400, dist="mse").to(device)
        coarse_critic = port.MLP(coarse_dim, [], 4, 400, dist="mse").to(device)
        ac_opt = torch.optim.Adam([*actor.parameters(), *critic.parameters(), *coarse_critic.parameters()], 1e-4)

        def wm_step():
            loss, _, outs, _ = wm.loss(data, noise=noise)
            model_opt.step(loss)
            return outs

        def thick_step():
            total, _, _ = wm.thick_loss(data, noise=noise)
            thick_opt.step(total)

        def behaviour_step(outs):
            with torch.no_grad():
                state = {k: v.reshape(-1, *v.shape[2:]).detach() for k, v in outs["post"].items()}
                feats, coarse = [wm.rssm.get_feat(state)], [wm.rssm.get_coarse_pred(state)]
                for _ in range(15):
                    action = actor(feats[-1]).sample(noise).detach()
                    state = wm.rssm.img_step(state, action, noise=noise)
                    feats.append(wm.rssm.get_feat(state))
                    coarse.append(wm.rssm.get_coarse_pred(state))
                hl = {k: v for k, v in state.items()}
                for _ in range(16):
                    pred = wm.thick_net.sample_step(port.gen_thick_inps(hl["context"], hl["stoch"]), noise=noise)
                    hl = wm.rssm.img_step({"stoch": pred["thick_stoch"], "context": hl["context"],
                                           "deter": hl["deter"]}, pred["thick_act"], noise=noise)
                    coarse.append(wm.rssm.get_coarse_pred(hl))
            feats, coarse = torch.stack(feats), torch.stack(coarse)
            policy = actor(feats[:-2])
            actions = torch.nn.functional.one_hot(torch.randint(0, 5, feats[:-2].shape[:-1], device=device), 5)
            target = torch.zeros(feats.shape[:2], device=device)
            loss = -(policy.log_prob(actions.float()) + 2e-3 * policy.entropy()).mean()
            loss = loss - critic(feats[:-1]).log_prob(target[:-1]).mean()
            loss = loss - coarse_critic(coarse[:16]).log_prob(target[:16]).mean()
            ac_opt.zero_grad()
            loss.backward()
            ac_opt.step()

        def act_step():
            with torch.no_grad():
                one = {k: v[:1, :1] for k, v in data.items()}
                embed = wm.encoder(wm.preprocess(one))[:, 0]
                state, _ = wm.rssm.obs_step(wm.rssm.initial(1, device), torch.zeros(1, 5, device=device), embed,
                                            one["is_first"][:, 0], noise=noise)
                actor(wm.rssm.get_feat(state)).sample(noise)

        outs = wm_step()
        return {
            "batch": [b, t], "world_model_parameters": sum(p.numel() for p in wm.parameters()),
            "seconds_wm_update": _timed(wm_step, 1, repeats, sync),
            "seconds_thick_update": _timed(thick_step, 1, repeats, sync),
            "seconds_behaviour_proxy": _timed(lambda: behaviour_step(outs), 1, repeats, sync),
            "seconds_act_step_batch1": _timed(act_step, 3, 50, sync),
            "peak_memory_gb": peak()}

    for tf32 in (False, True):
        result[f"pinpad_image_{'tf32' if tf32 else 'fp32'}"] = measure(tf32)
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    # NovPhy carrier config (the D1 arm), TF32 off
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wm = nv.build_model().to(device)
    carriers = torch.randn(nv.BATCH, len(nv.POSITIONS), nv.CARRIER_DIM, generator=gen, device=device)
    launch = torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]]).repeat(nv.BATCH, 1)
    batch = {k: v.to(device) for k, v in nv.episode_batch(carriers.cpu(), launch).items()}
    model_opt, thick_opt = wm.make_optimizers()
    per_step = _timed(lambda: wm.train_step(batch, model_opt, thick_opt, noise=noise), 2, repeats, sync)
    anchors = torch.randn(16, nv.CARRIER_DIM, device=device)
    launches = launch.to(device)
    rollout = _timed(lambda: nv.rollout(wm, anchors, launches), 1, max(1, repeats // 2), sync)
    result["novphy_carrier_fp32"] = {"batch": [nv.BATCH, len(nv.POSITIONS)], "parameters": nv.parameter_count(wm),
                                     "seconds_per_wm_and_thick_update": per_step,
                                     "seconds_per_16_candidate_rollout": rollout,
                                     "peak_memory_gb": peak()}
    write_json(out_json, result)
    print(json.dumps(result, indent=1), flush=True)
    return 0


# ---------------------------------------------------------------------------
# the G-fid plan and its projection
# ---------------------------------------------------------------------------

PLAN_CONSTANTS = {"train_every": 5, "eval_every": 1e4, "eval_episode_steps": 2000, "p2e_steps": 1e6,
                  "overhead_fraction": 0.15, "p2e_update_over_dreamer": 1.15,
                  "window_steps": {"four": 4e5, "five": 6e5}}
PROPOSAL = {
    "route": "(ii) the G-port-verified PyTorch port extended to the full THICK Dreamer agent, DreamerV2 and "
             "Plan2Explore; every added module held to G-port against the vendored TF2 code on CPU before any run",
    "design": "four-seeds3-p2e1",
    "task": "pinpad_four (VisualPinPadFour: the paper's tuning task; band threshold 359, Table 3 140k vs 200k)",
    "env_steps": 400_000, "seeds_per_method": [20260908, 20260909, 20260910],
    "exploration": "one Plan2Explore run (upstream train.py, configs visualpinpad + plan2explore, 1M env steps) whose "
                   "replay pre-fills every THICK Dreamer / DreamerV2 run (upstream train_from_replay.py); the paper "
                   "trained two Plan2Explore models and kept the higher-return one (disclosed)",
    "evaluation": "one evaluation episode (2000 steps) every 1e4 env steps (upstream eval_every / eval_eps)",
    "precision": "float32, TF32 off (as the G-port and the D1 adapter); the official TF 2.4 itself enables TF32 on "
                 "this GPU (its log), TF32 would cut the projection by about a quarter",
    "resumable": "SIGTERM -> checkpoint (models, optimizers, replay on disk, RNG, step) -> exit 75",
}
PORT_MODULES = [
    {"module": "EnsembleRSSM (DreamerV2 low level, DreamerV2 baseline and Plan2Explore)",
     "upstream": "common/nets.py EnsembleRSSM", "gport": "forward, KL, gradients, two Adam steps",
     "note": "the TF reference needs the listed one-line vendored fix (undefined rnn_type)"},
    {"module": "ActorCritic (actor onehot / reinforce, critic, slow target, StreamNorm, lambda return)",
     "upstream": "agent.py ActorCritic, common/other.py", "gport": "imagined sequence, losses, gradients, steps"},
    {"module": "WorldModel.imagine and THICK Dreamer's thick_imagine / long_term_return / coarse critic",
     "upstream": "agent.py", "gport": "imagined sequences with replayed draws, critic targets"},
    {"module": "Plan2Explore (10-head disagreement ensemble, intrinsic reward)",
     "upstream": "expl.py", "gport": "ensemble loss, intrinsic reward, gradients, steps"},
    {"module": "acting (Agent.policy), Replay (uniform 50-step chunks), Driver, PinPad + OneHotAction + TimeLimit",
     "upstream": "agent.py, common/{replay,driver,envs,pinpad_env}.py",
     "gport": "env transitions equal on fixed seeds and action sequences; replay chunk sampling semantics (CPU "
              "tests)"},
    {"module": "train.py (Plan2Explore data) and train_from_replay.py (THICK Dreamer / DreamerV2) loops with "
               "resumable checkpoints (SIGTERM -> checkpoint -> exit 75)",
     "upstream": "train.py, train_from_replay.py", "gport": "schedule equality (prefill, train_every, eval_every, "
                                                             "eval_eps) by CPU tests"},
]


def _read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def plan():
    timing = _read(FIDELITY / "port-timing-gpu-r2.json")
    band_doc = _read(FIDELITY / "band.json")
    if timing is None or band_doc is None:
        raise FileNotFoundError("plan needs fidelity/port-timing-gpu-r2.json and fidelity/band.json")
    c = PLAN_CONSTANTS
    projections = {}
    for precision in ("fp32", "tf32"):
        t = timing[f"pinpad_image_{precision}"]
        act = t["seconds_act_step_batch1"]
        updates = {"thick_dreamer": t["seconds_wm_update"] + t["seconds_thick_update"] + t["seconds_behaviour_proxy"],
                   "dreamerv2": t["seconds_wm_update"] + t["seconds_behaviour_proxy"]}
        updates["plan2explore"] = updates["dreamerv2"] * c["p2e_update_over_dreamer"]

        def run_hours(kind, steps):
            evals = steps / c["eval_every"] + 1
            seconds = steps / c["train_every"] * updates[kind] + (steps + evals * c["eval_episode_steps"]) * act
            return seconds * (1 + c["overhead_fraction"]) / 3600

        per_run = {f"{kind}_{task}": run_hours(kind, steps) for task, steps in c["window_steps"].items()
                   for kind in ("thick_dreamer", "dreamerv2")}
        per_run["plan2explore_1M"] = run_hours("plan2explore", c["p2e_steps"])
        designs = {}
        for task in ("four", "five"):
            for seeds in (3, 5, 7):
                for p2e in (1, 2):
                    gpu_h = seeds * (per_run[f"thick_dreamer_{task}"] + per_run[f"dreamerv2_{task}"]) \
                        + p2e * per_run["plan2explore_1M"]
                    designs[f"{task}-seeds{seeds}-p2e{p2e}"] = round(gpu_h, 1)
        designs["four+five-seeds3-p2e1x2"] = round(designs["four-seeds3-p2e1"] + designs["five-seeds3-p2e1"], 1)
        projections[precision] = {"seconds_per_update": {k: round(v, 4) for k, v in updates.items()},
                                  "seconds_per_act_step": round(act, 5),
                                  "gpu_hours_per_run": {k: round(v, 2) for k, v in per_run.items()},
                                  "gpu_hours_by_design": designs}
    carrier = timing["novphy_carrier_fp32"]
    native_hours = 200_000 * carrier["seconds_per_wm_and_thick_update"] * (1 + c["overhead_fraction"]) / 3600
    payload = {
        "schema": "issue_131_gfid_plan_v1", "issue": 131, "arm": "D1 THICK",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "band": {"artifact": str((FIDELITY / "band.json").relative_to(ROOT)),
                 "sha256": sha256_of(FIDELITY / "band.json")},
        "official_tf_on_rtx5090": {
            "tf242_round1": _read(FIDELITY / "tf-gpu-probe-tf242.json"),
            "tf242_round2": _read(FIDELITY / "tf-gpu-probe-tf242-r2.json"),
            "tf242_log_summary": _read(FIDELITY / "tf242-probe-summary.json"),
            "tf2x_round1": _read(FIDELITY / "tf-gpu-probe-tf2x.json"),
            "tf2x_round2": _read(FIDELITY / "tf-gpu-probe-tf2x-r2.json"),
            "paper_wall_clock": "Suppl. G: DreamerV2 runs 15-20 h, THICK Dreamer runs 35-45 h per experiment on "
                                "one GPU of the authors' cluster (official TF2 code)"},
        "port_timing": {"artifact": str((FIDELITY / "port-timing-gpu-r2.json").relative_to(ROOT)),
                        "sha256": sha256_of(FIDELITY / "port-timing-gpu-r2.json")},
        "projection_model": {
            **c, "formula": "GPU-h per run = (steps / train_every * s_update + (steps + evals * 2000) * s_act) * "
                            "(1 + overhead) / 3600; s_update(THICK Dreamer) = world model + high level + behaviour "
                            "proxy; s_update(DreamerV2) = C-RSSM world model + behaviour proxy (an upper bound: "
                            "the RSSM has no context decoder / head and DreamerV2 no high-level imagination); "
                            "Plan2Explore = DreamerV2 x 1.15 [INFERENCE]",
            "status": "[INFERENCE] until the first G-fid run's measured throughput replaces it"},
        "projections": projections,
        "port_modules_to_add": PORT_MODULES,
        "novphy_native_training_gpu_hours_per_seed": round(native_hours, 2),
        "proposal": {
            **PROPOSAL,
            "gfid_gpu_hours": projections["fp32"]["gpu_hours_by_design"][PROPOSAL["design"]],
            "gfid_gpu_hours_tf32": projections["tf32"]["gpu_hours_by_design"][PROPOSAL["design"]],
            "child_projection_gpu_hours": {
                "fidelity": projections["fp32"]["gpu_hours_by_design"][PROPOSAL["design"]],
                "selection_3_configurations": round(3 * native_hours, 1),
                "training_2_more_seeds": round(2 * native_hours, 1),
                "matched_secondary_3_seeds": 2.0, "scoring_and_validate": 2.0},
            "child_share_gpu_hours": [86, 146], "fidelity_estimate_gpu_hours": [24, 48]},
    }
    write_json(FIDELITY / "plan.json", payload)
    print(json.dumps({k: payload[k] for k in ("projections", "novphy_native_training_gpu_hours_per_seed")},
                     indent=1), flush=True)
    return 0


# ---------------------------------------------------------------------------
# G-fid timing (first 1000 updates of each kind) and the G-fid record
# ---------------------------------------------------------------------------

GFID_RUNS = {"plan2explore": [20260908], "thick": [20260908, 20260909, 20260910],
             "dreamer": [20260908, 20260909, 20260910]}
GFID_STEPS = {"plan2explore": 1_000_000, "thick": 400_000, "dreamer": 400_000}
GFID_FIDELITY_CAP_GPU_H = 146 - (42.0 + 28.0 + 2.0 + 2.0)  # child share minus the rest of the projected child


def run_dir(kind, seed, task="pinpad_four"):
    return FIDELITY / "runs" / f"{kind}-{task}-seed{seed}"


def gfid_timing(updates=1000, device="cuda", base=None):
    """Timing-only runs (``updates`` updates each, separate logdirs, exit 0 without checkpoints) of the three kinds,
    then the re-checked G-fid projection; exit 0 iff it fits ``GFID_FIDELITY_CAP_GPU_H``."""
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.thick import fidelity_runner

    base = Path(base) if base else FIDELITY / "timing"
    timings = {}
    for kind in ("plan2explore", "thick", "dreamer"):
        logdir = base / kind
        shutil.rmtree(logdir, ignore_errors=True)  # a timing run always starts fresh
        fidelity_runner.STOP.clear()
        status = fidelity_runner.main(kind, 20260908, logdir, load_dir=base / "plan2explore" if kind != "plan2explore"
                                      else None, stop_after_updates=updates + 2, timing_updates=updates, device=device,
                                      ledger_dir=OUTPUT / "ledger", run_name=f"gfid-timing-{kind}")
        if status != 0:
            raise RuntimeError(f"timing run {kind} ended with {status}")
        timings[kind] = json.loads((logdir / "timing.json").read_text())
    per_run = {}
    for kind, t in timings.items():
        act = max(t["wall_seconds_per_env_step"] - t["seconds_per_update"] / 5, 0.0)
        steps = GFID_STEPS[kind]
        evals = steps / 1e4 + 1
        per_run[kind] = (steps * t["wall_seconds_per_env_step"] + evals * 2000 * act) / 3600
    total = sum(per_run[k] * len(GFID_RUNS[k]) for k in per_run)
    payload = {"schema": "issue_131_gfid_timing_v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "timings": timings, "gpu_hours_per_run": {k: round(v, 2) for k, v in per_run.items()},
               "runs": GFID_RUNS, "gfid_gpu_hours": round(total, 1), "projected_gpu_h": round(total, 1),
               "projected_in_plan_gpu_hours": {"fp32": 67.2, "tf32": 51.4},
               "cap_gpu_hours": GFID_FIDELITY_CAP_GPU_H,
               "rule": "feasible iff the re-projected G-fid GPU-h (measured wall seconds per env step over the first "
                       f"{updates} updates, plus 2000-step evaluation episodes every 1e4 env steps) fits the child "
                       "share minus the rest of the projected child (selection 42, training 28, matched 2, scoring 2)",
               "feasible": bool(total <= GFID_FIDELITY_CAP_GPU_H)}
    write_json(base.parent / "gfid-timing.json" if base != FIDELITY / "timing" else FIDELITY / "gfid-timing.json",
               payload)
    print(json.dumps({k: payload[k] for k in ("gpu_hours_per_run", "gfid_gpu_hours", "feasible")}), flush=True)
    return 0 if payload["feasible"] else 1


def gfid():
    """``fidelity/gfid.json`` (schema issue_127_gfid_v1) from the completed runs and the frozen band."""
    import numpy as np

    band_doc = json.loads((FIDELITY / "band.json").read_text())
    task = band_doc["tasks"]["four"]
    curves, reports = {}, {}
    for kind in ("thick", "dreamer"):
        per_seed = {}
        for seed in GFID_RUNS[kind]:
            d = run_dir(kind, seed)
            reports[f"{kind}-{seed}"] = json.loads((d / "report.json").read_text()) if (d / "report.json").is_file() \
                else None
            rows = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines()] \
                if (d / "metrics.jsonl").is_file() else []
            per_seed[seed] = {int(r["step"]): r["eval_return"] for r in rows if "eval_return" in r}
        steps = sorted(set.intersection(*(set(v) for v in per_seed.values()))) if all(per_seed.values()) else []
        curves[kind] = {"steps": steps, "seed_mean": [float(np.mean([per_seed[s][t] for s in per_seed]))
                                                      for t in steps],
                        "per_seed": {str(s): v for s, v in per_seed.items()}}
    complete = all(reports.values())
    result = {"schema": "issue_127_gfid_v1", "issue": 131, "arm": "D1 THICK",
              "band": {"artifact": str((FIDELITY / "band.json").relative_to(ROOT)),
                       "sha256": sha256_of(FIDELITY / "band.json")},
              "task": "pinpad_four", "window_steps": 400_000, "threshold": task["threshold"],
              "complete": complete, "reports": reports, "curves": curves}
    if complete:
        out = {}
        for kind, band_key in (("thick", "thick"), ("dreamer", "dreamer")):
            c = curves[kind]
            final = dict(zip(c["steps"], c["seed_mean"])).get(400_000)
            lo, hi = task[band_key]["band"]
            crossing = next((s for s, m in zip(c["steps"], c["seed_mean"]) if m >= task["threshold"]), None)
            out[kind] = {"final_seed_mean": final, "band": [lo, hi],
                         "in_band": final is not None and lo <= final <= hi, "steps_to_threshold": crossing,
                         "paper_final": task[band_key]["mean"],
                         "paper_steps_to_threshold": task[band_key]["steps_to_threshold"]}
        t, d = out["thick"]["steps_to_threshold"], out["dreamer"]["steps_to_threshold"]
        advantage = t is not None and (d is None or t < d)
        result.update({"methods": out, "advantage": advantage,
                       "pass": bool(out["thick"]["in_band"] and out["dreamer"]["in_band"] and advantage)})
    else:
        result["pass"] = False
    write_json(FIDELITY / "gfid.json", result)
    print(json.dumps({k: result.get(k) for k in ("complete", "advantage", "pass")}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--band", action="store_true")
    group.add_argument("--plan", action="store_true")
    group.add_argument("--tf-gpu-probe", metavar="OUT_JSON")
    group.add_argument("--port-timing", metavar="OUT_JSON")
    group.add_argument("--gfid-timing", action="store_true", help="timing-only runs + re-checked projection")
    group.add_argument("--gfid", action="store_true", help="write fidelity/gfid.json from the completed runs")
    group.add_argument("--build-cache", metavar="LOAD_DIR", help="build the exploration replay cache once (CPU)")
    group.add_argument("--verify-cache", metavar="LOAD_DIR", help="verify the cache against the episodes (CPU)")
    group.add_argument("--run", choices=["plan2explore", "thick", "dreamer"],
                       help="a G-fid run (resumable: SIGTERM -> checkpoint -> exit 75)")
    parser.add_argument("--official-timing", action="store_true")
    parser.add_argument("--device", default="cuda", help="--port-timing / --run device (cpu for CPU tests)")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--task", default="pinpad_four")
    parser.add_argument("--steps", type=int, default=None, help="--run: env steps (default: upstream config.steps)")
    parser.add_argument("--load-dir", default=None, help="--run thick / dreamer: the exploration run's logdir")
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--stop-after-updates", type=int, default=None,
                        help="--run: a timing-only run that ends (exit 0, no checkpoint) after this many updates")
    args = parser.parse_args(argv)
    if args.band:
        return band()
    if args.plan:
        return plan()
    if args.tf_gpu_probe:
        return tf_gpu_probe(Path(args.tf_gpu_probe).resolve(), args.official_timing)
    if args.gfid_timing:
        return gfid_timing()
    if args.gfid:
        return gfid()
    if args.build_cache or args.verify_cache:
        sys.path.insert(0, str(ROOT))
        from world_model.baselines.thick import envs
        from world_model.baselines.thick.fidelity_runner import envs_cache_root
        load_dir = Path(args.build_cache or args.verify_cache)
        source, cache = load_dir / "train_episodes", envs_cache_root(load_dir)
        if args.build_cache:
            t0 = time.time()
            manifest = envs.build_cache(source, cache)
            print(f"[b127-thick-cache] built {cache}: {len(manifest['episodes'])} episodes, {manifest['steps']} steps, "
                  f"{time.time() - t0:.0f} s", flush=True)
            return 0
        t0 = time.time()
        result = envs.verify_cache(source, cache)
        result.update({"schema": "issue_131_replay_cache_verification_v1", "source": str(source), "cache": str(cache),
                       "manifest_sha256": sha256_of(cache / "MANIFEST.json"), "seconds": round(time.time() - t0, 1),
                       "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        write_json(FIDELITY / "cache-verification.json", result)
        print(json.dumps({k: result[k] for k in ("episodes", "arrays", "steps", "pass")}), flush=True)
        return 0 if result["pass"] else 1
    if args.run:
        sys.path.insert(0, str(ROOT))
        from world_model.baselines.thick import fidelity_runner
        logdir = Path(args.logdir) if args.logdir else FIDELITY / "runs" / f"{args.run}-{args.task}-seed{args.seed}"
        timing_only = args.stop_after_updates is not None
        return fidelity_runner.main(args.run, args.seed, logdir, task=args.task, load_dir=args.load_dir,
                                    steps=args.steps, device=args.device, tf32=not args.no_tf32,
                                    stop_after_updates=args.stop_after_updates,
                                    ledger_dir=OUTPUT / "ledger",
                                    run_name=f"gfid-{'timing-' if timing_only else ''}{args.run}-{args.task}-"
                                             f"seed{args.seed}")
    return port_timing(Path(args.port_timing).resolve(), args.device, args.repeats)


if __name__ == "__main__":
    sys.exit(main())
