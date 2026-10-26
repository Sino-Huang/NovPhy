"""#137 G-port (Decision P-route on #137): the official DIAMOND code (eloialonso/diamond ``main`` @ 5bcd159, vendored,
run from ``vendor/src`` in env ``b127-diamond``) against the NovPhy adapter (``world_model/baselines/diamond/
novphy.py``, env ``novphy``) on fixed CPU float32 inputs, both on torch 2.13.0+cu130.

Modes (in order):
1. ``--inputs`` (novphy): 8 fit shots (64 x 64 frames at stride 5, launches; the runner's frame cache), fixed window
   starts (with left padding and the launch frame), 2 fit-anchor frames + 2 grid launches for the rollout, and a
   synthetic Atari-shaped batch -> ``gport/inputs.pt``.
2. ``--reference`` (b127-diamond): per case, the official objects built from ``config/agent/default.yaml`` /
   ``config/trainer.yaml`` with ``hydra.utils.instantiate``: initial state (torch seed), ``Denoiser.forward`` (the
   loss), two optimizer steps by ``Trainer.train_component`` itself (``utils.configure_opt`` / ``get_lr_sched``;
   pre-clip gradient norm, post-clip gradients and parameters after each step captured around ``opt.step``), one
   ``DiffusionSampler.sample`` call, and (``launch``) a 45-step ``WorldModelEnv.step`` rollout (stub reward / end model
   that never ends an episode; torch re-seeded before each ``predict_next_obs``). The NovPhy batches are built by the
   official ``Episode`` / ``make_segment`` / ``collate_segments_to_batch``. Cases: ``atari-discrete`` (the unchanged
   official model, 6 discrete actions, schedulers advanced to the full LR), ``launch`` (num_actions 5, the action
   table applied as the csgo branch does: ``act.float() @ act_emb[0].weight``; warm-up steps 0 and 1),
   ``launch-ar4`` (num_autoregressive_steps 4) -> ``gport/reference.pt``.
3. ``--compare`` (novphy): the adapter on the same inputs and seeds -> ``gport/gport.json`` (schema
   ``issue_127_gport_v1``; #123 tolerances; expected exact on the same torch build).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "world_model" / "baselines" / "diamond" / "vendor"
OUTPUT = ROOT / ".local-artifacts" / "issue-137-diamond-v1"
INIT_SEED, FWD_SEED, STEP_SEED, SAMPLE_SEED, ROLL_SEED = 1370, 1371, 1372, 1373, 1374
ATARI_ACTIONS = 6
BATCH = 4
STARTS = {"ar1": [-2, 0, 3, 43], "ar4": [-4, 0, 3, 40]}  # window starts (episode = 49 frames, launch at index 3)
ROLLOUT_STEPS = 45
CASES = {"atari-discrete": {"ar": 1, "lr_offset": 100}, "launch": {"ar": 1, "lr_offset": 0},
         "launch-ar4": {"ar": 4, "lr_offset": 0}}
TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
             "param_close_abs": 1e-7, "param_close_share": 0.999, "sample_max_abs": 1e-5}


def sha256(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------
# --inputs (novphy)
# ---------------------------------------------------------------------------------------------------------------

def make_inputs(output: Path) -> None:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from scripts import external_baselines as eb
    from scripts import run_issue_137_diamond as r137
    from world_model.baselines.diamond import novphy as dn
    shots = eb.FitShots()
    data = r137.load_data(output, shots)
    picks = [0, 1, 2, 3, 1000, 5000, 9000, 13000]
    frames = torch.from_numpy(np.stack([np.asarray(data["frames"][s]) for s in picks]))
    launch5 = data["launch5"][picks].clone()
    anchors = r137.load_fit_anchor_frames(output)
    members = sorted(anchors)[:2]
    items, actions = r137.grid_actions()
    g = torch.Generator().manual_seed(INIT_SEED)
    atari_obs = torch.randint(0, 256, (2, BATCH, 6, 3, dn.IMAGE, dn.IMAGE), generator=g, dtype=torch.uint8)
    atari = {"obs": atari_obs.float().div(255).mul(2).sub(1),
             "act": torch.randint(0, ATARI_ACTIONS, (2, BATCH, 6), generator=g),
             "mask": torch.ones(2, BATCH, 6, dtype=torch.bool)}
    atari["mask"][:, 0, :2] = False  # one left-padded window per batch
    atari["obs"][:, 0, :2] = 0
    atari["act"][:, 0, :2] = 0
    payload = {"schema": "issue_137_gport_inputs_v1", "frames": frames, "launch5": launch5, "shots": picks,
               "starts": STARTS, "episode_frames": list(dn.EPISODE_FRAMES), "launch_index": dn.LAUNCH_INDEX,
               "rollout_frames": torch.stack([anchors[m] for m in members]), "rollout_members": members,
               "rollout_launch5": actions[[0, 15]].clone(), "rollout_steps": ROLLOUT_STEPS, "atari": atari,
               "written_at": utc_now()}
    (output / "gport").mkdir(parents=True, exist_ok=True)
    torch.save(payload, output / "gport" / "inputs.pt")
    print(f"inputs: {len(picks)} shots, rollout members {members}")


def windows(inputs, case_ar, step):
    """(shot indices into inputs['frames'], starts) of batch ``step`` (0 or 1) of a NovPhy case."""
    return list(range(step * BATCH, (step + 1) * BATCH)), STARTS["ar1" if case_ar == 1 else "ar4"]


# ---------------------------------------------------------------------------------------------------------------
# --reference (b127-diamond): the official code
# ---------------------------------------------------------------------------------------------------------------

def _official():
    sys.path.insert(0, str(VENDOR / "src"))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    import data as vdata
    import models.diffusion as vdiff
    import trainer as vtrainer
    import utils as vutils
    from envs import world_model_env as vwme
    agent_cfg = OmegaConf.load(VENDOR / "config/agent/default.yaml")
    trainer_cfg = OmegaConf.load(VENDOR / "config/trainer.yaml")
    return SimpleNamespace(instantiate=instantiate, OmegaConf=OmegaConf, data=vdata, diffusion=vdiff,
                           trainer=vtrainer, utils=vutils, wme=vwme, agent_cfg=agent_cfg, trainer_cfg=trainer_cfg)


def official_denoiser(o, num_actions, launch):
    cfg = o.instantiate(o.agent_cfg.denoiser)
    cfg.inner_model.num_actions = num_actions  # AgentConfig.__post_init__
    torch.manual_seed(INIT_SEED)
    model = o.diffusion.Denoiser(cfg)
    if launch:  # the csgo branch's real-valued action embedding: act.float() @ act_emb[0].weight
        table = model.inner_model.act_emb[0]
        table.forward = lambda act, table=table: act.float() @ table.weight
    model.setup_training(o.instantiate(o.trainer_cfg.denoiser.sigma_distribution))
    return model


def official_batches(o, inputs, case):
    """The two batches of a case as official ``Batch`` objects (NovPhy: Episode -> make_segment -> collate)."""
    if case == "atari-discrete":
        a = inputs["atari"]
        z = torch.zeros(BATCH, 6)
        return [o.data.Batch(a["obs"][s], a["act"][s], z, z.long(), z.long(), a["mask"][s], [{}] * BATCH, [])
                for s in range(2)]
    ar = CASES[case]["ar"]
    length = 4 + 1 + ar
    out = []
    for step in range(2):
        idx, starts = windows(inputs, ar, step)
        segments = []
        for i, start in zip(idx, starts):
            frames = inputs["frames"][i][inputs["episode_frames"]].float().div(255).mul(2).sub(1)
            act = torch.zeros(len(inputs["episode_frames"]), 5)
            act[inputs["launch_index"]] = inputs["launch5"][i]
            n = len(frames)
            episode = o.data.Episode(frames, act, torch.zeros(n), torch.zeros(n, dtype=torch.uint8),
                                     torch.zeros(n, dtype=torch.uint8), {})
            segments.append(o.data.make_segment(episode, o.data.SegmentId(i, start, start + length), should_pad=True))
        out.append(o.data.collate_segments_to_batch(segments))
    return out


class _Tools(SimpleNamespace):
    def get(self, name):
        return getattr(self, name)

    def set(self, name, value):
        setattr(self, name, value)


def official_steps(o, model, batches, lr_offset):
    """Two optimizer steps by the official ``Trainer.train_component`` (captures around ``opt.step``)."""
    ocfg = o.trainer_cfg.denoiser.optimizer
    opt = o.utils.configure_opt(model, ocfg.lr, ocfg.weight_decay, ocfg.eps)
    sched = o.utils.get_lr_sched(opt, o.trainer_cfg.denoiser.training.lr_warmup_steps)
    for _ in range(lr_offset):
        sched.step()
    captured = []
    step = opt.step

    def wrapped(*args, **kwargs):
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        out = step(*args, **kwargs)
        captured.append({"grads": grads, "params": {n: p.detach().clone() for n, p in model.named_parameters()}})
        return out
    opt.step = wrapped
    cfg = o.OmegaConf.create({"denoiser": {"training": {"grad_acc_steps": 1,
                                                        "max_grad_norm": o.trainer_cfg.denoiser.training.max_grad_norm}}})
    stub = SimpleNamespace(_cfg=cfg, _agent=SimpleNamespace(denoiser=model), opt=_Tools(denoiser=opt),
                           lr_sched=_Tools(denoiser=sched), _data_loader_train=_Tools(denoiser=batches),
                           num_batch_train=_Tools(denoiser=0), _rank=1, _device=torch.device("cpu"))
    torch.manual_seed(STEP_SEED)
    logs = o.trainer.Trainer.train_component(stub, "denoiser", 2)
    steps = []
    for k, metrics in enumerate(logs):
        steps.append({"loss": float(metrics["denoiser/train/loss_denoising"]),
                      "grad_norm_preclip": float(metrics["denoiser/train/grad_norm_before_clip"]),
                      "lr": float(metrics["denoiser/train/lr"]), **captured[k]})
    return steps


class _StubRewEnd:
    """Reward / end model stand-in: never ends an episode (end logits [0, -1e9]); LSTM state of size 1."""

    def predict_rew_end(self, obs, act, next_obs, hx_cx=None):
        b, t = obs.shape[:2]
        rew = torch.zeros(b, t, 3)
        end = torch.tensor([0.0, -1e9]).expand(b, t, 2).clone()
        return rew, end, (torch.zeros(1, b, 1), torch.zeros(1, b, 1))


def official_rollout(o, model, inputs):
    """45 ``WorldModelEnv.step`` calls from the decision frame x4 (actions [0, 0, 0, launch], no-ops after); torch is
    re-seeded with ROLL_SEED + k before the k-th ``predict_next_obs`` (the sampler's only random draw)."""
    frames0 = inputs["rollout_frames"].float().div(255).mul(2).sub(1)
    b = frames0.shape[0]
    ctx = o.data.Batch(frames0[:, None].repeat(1, 4, 1, 1, 1), torch.zeros(b, 4, 5), torch.zeros(b, 4),
                       torch.zeros(b, 4).long(), torch.zeros(b, 4).long(), torch.ones(b, 4, dtype=torch.bool),
                       [{}] * b, [])
    loader = SimpleNamespace(batch_sampler=SimpleNamespace(batch_size=b), __iter__=None)

    class _Loader:
        batch_sampler = loader.batch_sampler

        def __iter__(self):
            while True:
                yield ctx
    wcfg = o.instantiate(o.trainer_cfg.world_model_env)
    wcfg.horizon = 10_000
    wcfg.num_batches_to_preload = 1
    env = o.wme.WorldModelEnv(model, _StubRewEnd(), _Loader(), wcfg)
    env.reset()
    predict = env.predict_next_obs
    counter = {"k": 0}

    def seeded():
        torch.manual_seed(ROLL_SEED + counter["k"])
        counter["k"] += 1
        return predict()
    env.predict_next_obs = seeded
    out = []
    for k in range(inputs["rollout_steps"]):
        act = inputs["rollout_launch5"] if k == 0 else torch.zeros(b, 5)
        obs, *_ = env.step(act)
        out.append(obs.clone())
    return torch.stack(out)


@torch.no_grad()
def official_sample(o, model, batch):
    sampler = o.diffusion.DiffusionSampler(model, o.instantiate(o.trainer_cfg.world_model_env.diffusion_sampler))
    torch.manual_seed(SAMPLE_SEED)
    x, trajectory = sampler.sample(batch.obs[:, :4], batch.act[:, :4])
    return x


def reference(output: Path) -> None:
    torch.set_num_threads(4)
    o = _official()
    inputs = torch.load(output / "gport" / "inputs.pt", map_location="cpu", weights_only=False)
    result = {"schema": "issue_137_gport_reference_v1", "torch": torch.__version__, "cases": {}}
    for case, spec in CASES.items():
        discrete = case == "atari-discrete"
        model = official_denoiser(o, ATARI_ACTIONS if discrete else 5, not discrete)
        init = {k: v.clone() for k, v in model.state_dict().items()}
        batches = official_batches(o, inputs, case)
        torch.manual_seed(FWD_SEED)
        loss, _ = model(batches[0])
        model.zero_grad()
        steps = official_steps(o, model, batches, spec["lr_offset"])
        entry = {"init": init, "forward_loss": float(loss.detach()), "steps": steps,
                 "batch0": {"obs": batches[0].obs, "act": batches[0].act, "mask": batches[0].mask_padding},
                 "sample": official_sample(o, model, batches[1]),
                 "parameters": sum(p.numel() for p in model.parameters())}
        if case == "launch":
            entry["rollout"] = official_rollout(o, model, inputs)
        result["cases"][case] = entry
        print(f"reference {case}: loss {entry['forward_loss']:.6f}, step losses "
              f"{[round(s['loss'], 6) for s in steps]}, lr {[s['lr'] for s in steps]}")
    result["written_at"] = utc_now()
    torch.save(result, output / "gport" / "reference.pt")


# ---------------------------------------------------------------------------------------------------------------
# --compare (novphy): the adapter
# ---------------------------------------------------------------------------------------------------------------

def adapter_denoiser(dn, case):
    torch.manual_seed(INIT_SEED)
    if case == "atari-discrete":  # the unchanged official model (vendored classes), as the adapter loads them
        v = dn.vendor()
        a = dn.AGENT
        model = v.Denoiser(v.DenoiserConfig(inner_model=v.InnerModelConfig(**a["inner_model"],
                                                                          num_actions=ATARI_ACTIONS),
                                            sigma_data=a["sigma_data"], sigma_offset_noise=a["sigma_offset_noise"]))
        model.setup_training(v.SigmaDistributionConfig(**dn.SIGMA_DISTRIBUTION))
        return model
    return dn.build_denoiser()


def adapter_batches(dn, inputs, case):
    if case == "atari-discrete":
        a = inputs["atari"]
        z = torch.zeros(BATCH, 6)
        return [dn.vendor().Batch(a["obs"][s], a["act"][s], z, z.long(), z.long(), a["mask"][s], [{}] * BATCH, [])
                for s in range(2)]
    ar = CASES[case]["ar"]
    length = dn.seq_length({**dn.TRAINING, "num_autoregressive_steps": ar})
    out = []
    for step in range(2):
        idx, starts = windows(inputs, ar, step)
        out.append(dn.make_batch(inputs["frames"], inputs["launch5"], idx, starts, length))
    return out


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def compare(output: Path) -> int:
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.diamond import novphy as dn
    torch.set_num_threads(4)
    inputs = torch.load(output / "gport" / "inputs.pt", map_location="cpu", weights_only=False)
    ref = torch.load(output / "gport" / "reference.pt", map_location="cpu", weights_only=False)
    cases = {}
    for case, spec in CASES.items():
        r = ref["cases"][case]
        model = adapter_denoiser(dn, case)
        init_equal = all(torch.equal(v, r["init"][k]) for k, v in model.state_dict().items()) and \
            set(model.state_dict()) == set(r["init"])
        batches = adapter_batches(dn, inputs, case)
        batch_equal = bool(torch.equal(batches[0].obs, r["batch0"]["obs"]) and
                           torch.equal(batches[0].act.float(), r["batch0"]["act"].float()) and
                           torch.equal(batches[0].mask_padding, r["batch0"]["mask"]))
        torch.manual_seed(FWD_SEED)
        loss, _ = model(batches[0])
        forward = {"loss": [float(loss.detach()), r["forward_loss"]], "loss_rel": _rel(float(loss), r["forward_loss"])}
        forward["pass"] = forward["loss_rel"] <= TOLERANCE["loss_relative"]
        model.zero_grad()
        opt, sched = dn.build_optimizer(model)
        for _ in range(spec["lr_offset"]):
            sched.step()
        captured = []
        step_fn = opt.step

        def wrapped(*args, **kwargs):
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            out = step_fn(*args, **kwargs)
            captured.append({"grads": grads, "params": {n: p.detach().clone() for n, p in model.named_parameters()}})
            return out
        opt.step = wrapped
        model.train()
        opt.zero_grad()
        torch.manual_seed(STEP_SEED)
        terms = [dn.train_step(model, opt, sched, b) for b in batches]
        steps = []
        for k, (t, c, rs) in enumerate(zip(terms, captured, r["steps"])):
            names_equal = set(c["grads"]) == set(rs["grads"])
            grad_rel = max(float((c["grads"][n] - rs["grads"][n]).norm() / rs["grads"][n].norm().clamp_min(1e-12))
                           for n in rs["grads"]) if names_equal else None
            diffs = torch.cat([(c["params"][n] - rs["params"][n]).abs().flatten() for n in rs["params"]])
            close = float((diffs <= TOLERANCE["param_close_abs"]).float().mean())
            entry = {"index": k, "loss": [t["loss"], rs["loss"]], "loss_rel": _rel(t["loss"], rs["loss"]),
                     "lr": [t["lr"], rs["lr"]], "lr_equal": t["lr"] == rs["lr"],
                     "grad_norm_preclip": [t["grad_norm"], rs["grad_norm_preclip"]],
                     "grad_norm_rel": _rel(t["grad_norm"], rs["grad_norm_preclip"]), "grad_names_equal": names_equal,
                     "grad_max_rel_l2": grad_rel, "param_max_abs": float(diffs.max()), "param_close_share": close}
            entry["pass"] = bool(entry["loss_rel"] <= TOLERANCE["loss_relative"] and entry["lr_equal"] and
                                 entry["grad_norm_rel"] <= TOLERANCE["grad_norm_relative"] and names_equal and
                                 grad_rel <= TOLERANCE["grad_tensor_relative_l2"] and
                                 close >= TOLERANCE["param_close_share"])
            steps.append(entry)
        with torch.no_grad():
            sampler = dn.make_sampler(model)
            torch.manual_seed(SAMPLE_SEED)
            b1 = batches[1]
            x0 = torch.randn(b1.obs.shape[0], 3, dn.IMAGE, dn.IMAGE)
            x = dn.sample_next(sampler, b1.obs[:, :4], b1.act[:, :4], x0)
        sample = {"max_abs": float((x - r["sample"]).abs().max())}
        sample["pass"] = sample["max_abs"] <= TOLERANCE["sample_max_abs"]
        entry = {"init_state_equal": bool(init_equal), "batch0_equal": batch_equal, "forward": forward,
                 "steps": steps, "sample": sample, "parameters": [dn.trainable_parameters(model), r["parameters"]]}
        if case == "launch":
            frames0 = dn.to_unit(inputs["rollout_frames"])
            noise = []
            for k in range(inputs["rollout_steps"]):
                torch.manual_seed(ROLL_SEED + k)
                noise.append(torch.randn(frames0.shape[0], 3, dn.IMAGE, dn.IMAGE))
            keep = tuple(range(1, inputs["rollout_steps"] + 1))
            out = dn.rollout(model.eval(), frames0, inputs["rollout_launch5"], torch.stack(noise), keep=keep)
            pred = torch.stack([out[k] for k in keep])
            entry["rollout"] = {"steps": len(keep), "max_abs": float((pred - r["rollout"]).abs().max()),
                                "frame225_max_abs": float((pred[-1] - r["rollout"][-1]).abs().max())}
            entry["rollout"]["pass"] = entry["rollout"]["max_abs"] <= TOLERANCE["sample_max_abs"]
        entry["exact"] = bool(forward["loss_rel"] == 0 and all(s["param_max_abs"] == 0 for s in steps) and
                              sample["max_abs"] == 0 and entry.get("rollout", {"max_abs": 0})["max_abs"] == 0)
        entry["pass"] = bool(init_equal and batch_equal and forward["pass"] and all(s["pass"] for s in steps) and
                             sample["pass"] and entry.get("rollout", {"pass": True})["pass"])
        cases[case] = entry
        print(f"compare {case}: pass {entry['pass']} exact {entry['exact']}")
    record = {"schema": "issue_127_gport_v1", "issue": 137, "arm": "T5 DIAMOND-px",
              "G_port_rule": "#123 tolerances (loss / pre-clip grad-norm relative 1e-4, per-tensor post-clip gradient "
                             "relative L2 1e-3, parameters |delta| <= 1e-7 on >= 99.9 % after each of two optimizer "
                             "steps, LR equal) + initial state and the official batch (Episode / make_segment / "
                             "collate) equal + one sampler call and the 45-step WorldModelEnv rollout max abs <= 1e-5; "
                             "reference = the official code (Trainer.train_component, configure_opt, get_lr_sched, "
                             "DiffusionSampler.sample, WorldModelEnv.step) in env b127-diamond; same torch build",
              "tolerance": TOLERANCE, "torch": [torch.__version__, ref["torch"]], "cases": cases,
              "adapter_sha256": sha256(ROOT / "world_model/baselines/diamond/novphy.py"),
              "reference_script_sha256": sha256(Path(__file__)),
              "inputs_sha256": sha256(output / "gport" / "inputs.pt"),
              "reference_sha256": sha256(output / "gport" / "reference.pt"),
              "pass": all(c["pass"] for c in cases.values()), "written_at": utc_now()}
    write_json(output / "gport" / "gport.json", record)
    print(f"G-port pass {record['pass']}")
    return 0 if record["pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inputs", action="store_true")
    mode.add_argument("--reference", action="store_true")
    mode.add_argument("--compare", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.inputs:
        make_inputs(output)
        return 0
    if args.reference:
        reference(output)
        return 0
    return compare(output)


if __name__ == "__main__":
    raise SystemExit(main())
