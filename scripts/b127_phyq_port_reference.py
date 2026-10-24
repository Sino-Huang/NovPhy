"""#128 G-port for A3 PhyQ-RelDQN: the official Phy-Q relational DQN code vs the NovPhy adapter, CPU float32.

Two modes, two envs:

    # official side, conda env b127-phyq (torch 2.13 + torchvision 0.28; vendored Phy-Q code, cwd = vendor dir)
    CUDA_VISIBLE_DEVICES="" python scripts/b127_phyq_port_reference.py reference [out_dir]
    # NovPhy side, conda env novphy: replays the recorded inputs through world_model.baselines.phyq.novphy
    CUDA_VISIBLE_DEVICES="" python scripts/b127_phyq_port_reference.py compare [out_dir]

``out_dir`` defaults to ``.local-artifacts/issue-128-direct-predictors-v1/gport/a3``; ``compare`` writes
``gport.json`` (schema ``issue_127_gport_v1``) there.

**Official side** (vendored ``world_model/baselines/phyq/vendor``, phy-q/benchmark @ 514491c, no line changed), per arm
(``DQNRelationalSymbolic`` = A3-sym, ``DQNRelationalImage`` = A3-img; h 120, w 160, 180 outputs, if_save_local True):

- *Inputs*: 8 fit anchors (``issue-109-sc-fit-type01010{3,5}-00{1..4}-c00``). A3-sym: the official
  ``network.transform`` (``transform_full`` -> ``SymbolicStateReader.get_symbolic_image``, with the removed
  ``np.float`` alias set around it, VENDOR.json compatibility) on the engine GT JSON the adapter rebuilds
  (``novphy.gt_json``); stored as uint8 like the official memmap. A3-img: the official ``network.transform`` on the
  adapter's Phy-Q screenshot (``novphy.phyq_screenshot`` of frame 0 read as ``s99.image_tensor`` reads it); kept
  float (the official memmap would cast it to uint8, see ``upstream``). 4 grid launches per level (32 examples),
  actions from ``novphy.action_index``, fixed rewards, all terminal.
- *Network*: ``torch.manual_seed(20261024)`` then the official constructor (records the initial state).
- *Training*: the official outer loop of ``TrainLearningAgent.py`` twice (target net = a fresh official network loaded
  with the online weights, ``eval()``), each ``network.train_model_loader(target_net, ..., train_time=1, gamma=0.99,
  loader, optimizer=AdamW, lr)`` over a ``DataLoader(batch_size=16, shuffle=False, drop_last=True)`` of the 32
  examples -> 4 updates with a fresh AdamW after update 2; lr 3e-4, plus a separate 2-update run at lr 0 from the same
  initial state (the step-0 LR check). Instrumentation only: a forward hook (Q values), ``torch.nn.SmoothL1Loss``
  replaced during the call by a subclass recording its inputs and output, parameter hooks (pre-clamp gradients), an
  ``AdamW`` subclass recording post-clamp gradients, parameters, buffers and optimizer state around ``step()``.
- *Eval forward*: ``network.eval()`` Q values of the 32 inputs from the initial state (the scoring path).
- *Upstream bug demo*: ``DQNBase.train_model_memory`` (the ``--if_save_local False`` path) vs ``train_model_loader``
  on a toy ``DQNBase`` (``Linear(48, 3)``) and 8 terminal toy transitions: the recorded TD targets.

**Adapter side**: the same seed through ``novphy.build_model`` (initial state compared), the official initial state
loaded strictly, ``novphy.symbolic_state`` / ``image_state`` against the official transforms, ``make_optimizer``
(``resets=[2]``) + ``runner_train_step`` on the same batches, ``score_actions``-path eval forward.

Tolerances (#123 rule, as the A1 G-port): loss and gradient-norm relative 1e-4; per-tensor gradient relative L2 1e-3
above 1e-6 x the global norm (absolute below); parameters after each step: >= 99.9 % within 1e-7 and every element
within 2 x LR + 1e-7; BatchNorm buffers and AdamW moments allclose (rtol 1e-4, atol 1e-6 x max |.|); Q values max
abs 1e-4; inputs identical.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "world_model/baselines/phyq/vendor"
OUT = ROOT / ".local-artifacts/issue-128-direct-predictors-v1/gport/a3"
CAMPAIGN_FIT = ROOT / ".local-artifacts/issue-104-power-cohort-campaign-v1/issue109-fit/attempts"
SEED_INIT = 20261024
MEMBERS = [f"issue-109-sc-fit-type{family}-{i:03d}" for family in ("010103", "010105") for i in range(1, 5)]
GRID = [(x, y) for x in (-40, -80, -120, -160) for y in (5, 30, 55, 80)]  # eb.grid_items order (ordinal 0..15)
LAUNCHES_PER_LEVEL = 4
BATCH, LR = 16, 3e-4
REWARDS = [1, 0, 0, 0, 0, 1, 0, 0] * 4  # fixed toy verdicts (8 positives of 32)
ARMS = {"A3-sym": "DQNRelationalSymbolic", "A3-img": "DQNRelationalImage"}
TOL = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
       "grad_floor_of_global_norm": 1e-6, "param_close_share": 0.999, "param_close_abs": 1e-7,
       "param_max_abs_over_lr": 2.0, "buffer_rtol": 1e-4, "buffer_atol_of_max": 1e-6, "q_max_abs": 1e-4}


def sha256_of(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def clone_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def examples():
    """(member, grid ordinal) of the 32 examples: level i takes ordinals i, i+4, i+8, i+12 (mod 16)."""
    return [(m, (i + 4 * j) % 16) for i, m in enumerate(MEMBERS) for j in range(LAUNCHES_PER_LEVEL)]


def anchor_frame(attempt):
    """Frame 0 of the attempt's shot as ``s99.image_tensor`` reads it (uint8 [3, 240, 320])."""
    from PIL import Image
    root = Path(attempt) / "shot-1/observation-trace"
    record = json.loads((root / "observation_trace_manifest.json").read_text())["frame_records"][0]
    with Image.open(root / record["agent_observation"]["relative_path"]) as image:
        image = image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# official side (b127-phyq)
# ---------------------------------------------------------------------------

class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass


class Recorder:
    """Instrumentation shared by the hooks of one official run."""

    def __init__(self, network):
        self.network = network
        self.steps = []
        self.current = {}
        self.names = {id(p): n for n, p in network.named_parameters()}
        for name, param in network.named_parameters():
            param.register_hook(self._grad_hook(name))
        network.register_forward_hook(self._forward_hook)

    def _grad_hook(self, name):
        def hook(grad):
            self.current.setdefault("grads_pre", {})[name] = grad.detach().clone()
        return hook

    def _forward_hook(self, module, inputs, output):
        if module.training:
            self.current["q"] = output.detach().clone()

    def loss_class(self):
        recorder = self

        class SmoothL1Recorder(torch.nn.SmoothL1Loss):
            def forward(self, input, target):
                out = super().forward(input, target)
                recorder.current["loss"] = out.detach().mean().clone()
                recorder.current["target"] = target.detach().clone()
                recorder.current["q_taken"] = input.detach().clone()
                return out
        return SmoothL1Recorder

    def optimizer_class(self):
        recorder = self

        class RecordingAdamW(torch.optim.AdamW):
            def step(self, closure=None):
                net = recorder.network
                recorder.current["grads_post"] = {n: p.grad.detach().clone() for n, p in net.named_parameters()}
                recorder.current["lr"] = self.param_groups[0]["lr"]
                result = super().step(closure)
                recorder.current["params"] = {n: p.detach().clone() for n, p in net.named_parameters()}
                recorder.current["buffers"] = {n: b.detach().clone() for n, b in net.named_buffers()}
                recorder.current["adam"] = {recorder.names[id(p)]: {"exp_avg": s["exp_avg"].clone(),
                                                                     "exp_avg_sq": s["exp_avg_sq"].clone(),
                                                                     "step": float(s["step"])}
                                            for p, s in self.state.items()}
                recorder.steps.append(recorder.current)
                recorder.current = {}
                return result
        return RecordingAdamW


class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, states, actions, rewards):
        self.states, self.actions, self.rewards = states, actions, rewards

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        state = self.states[idx]
        return state, int(self.actions[idx]), state, float(self.rewards[idx]), True


def official_inputs(cls_name, network):
    from world_model.baselines.phyq import novphy as nv
    states, frames, gts = [], {}, {}
    per_member = {}
    for member in MEMBERS:
        attempt = CAMPAIGN_FIT / f"{member}-c00"
        if cls_name == "DQNRelationalSymbolic":
            physics = json.loads((attempt / "decision-physics-v1.json").read_text())
            gt = nv.gt_json(physics, nv.engine_colormaps(attempt)[1])
            with nv.numpy_float_alias():
                image = network.transform(gt)  # transform_full (if_save_local=True)
            per_member[member] = torch.from_numpy(np.asarray(image).astype(np.uint8))  # official uint8 memmap
            gts[member] = gt
        else:
            frame = anchor_frame(attempt)
            frames[member] = frame
            per_member[member] = network.transform(nv.phyq_screenshot(frame)).float()
    for member, _ in examples():
        states.append(per_member[member])
    actions = [nv.action_index(*GRID[ordinal]) for _, ordinal in examples()]
    return per_member, torch.stack(states), actions, frames


def official_train(cls, network, states, actions, lr, outer_calls):
    from torch.utils.data import DataLoader
    recorder = Recorder(network)
    loader = DataLoader(ExampleSet(states.numpy(), actions, REWARDS), batch_size=BATCH, shuffle=False,
                        drop_last=True, num_workers=0)
    original = torch.nn.SmoothL1Loss
    torch.nn.SmoothL1Loss = recorder.loss_class()
    try:
        for call in range(outer_calls):  # TrainLearningAgent.py: target net per outer step, then train_model_loader
            target_net = cls(h=120, w=160, outputs=180, device="cpu", if_save_local=True)
            target_net.load_state_dict(network.state_dict())
            target_net.eval()
            network.train_model_loader(target_net, total_train_time=call, train_time=1, gamma=0.99, loader=loader,
                                       optimizer=recorder.optimizer_class(), lr=lr)
    finally:
        torch.nn.SmoothL1Loss = original
    return recorder.steps


def upstream_bug_demo():
    """train_model_memory (long-indexed is_done) vs train_model_loader (bool mask) on 8 terminal toy transitions."""
    import random
    from torch.utils.data import DataLoader
    from world_model.baselines.phyq.vendor.LearningAgents.Memory import ReplayMemory
    from world_model.baselines.phyq.vendor.LearningAgents.RLNetwork.DQNBase import DQNBase

    class ToyQ(DQNBase):
        def __init__(self):
            super().__init__(h=2, w=2, device="cpu", outputs=3, if_save_local=False, writer=DummyWriter())
            self.input_type = "symbolic"
            self.lin = torch.nn.Linear(48, 3)

        def forward(self, x):
            return self.lin(x.flatten(1))

    torch.manual_seed(0)
    random.seed(0)
    states = [torch.rand(12, 2, 2).numpy() for _ in range(8)]
    rewards = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    actions = [0, 1, 2, 0, 1, 2, 0, 1]
    results = {}
    for path in ("memory", "loader"):
        net = ToyQ()
        torch.manual_seed(1)
        net.lin.reset_parameters()
        target = ToyQ()
        target.load_state_dict(net.state_dict())
        captured = []
        pre = []
        net.register_forward_pre_hook(lambda m, inp: pre.append(inp[0].detach().clone()))

        class Capture(torch.nn.SmoothL1Loss):
            def forward(self, input, target_):
                captured.append((input.detach().clone(), target_.detach().clone(), super().forward(input, target_)))
                return super().forward(input, target_)
        original = torch.nn.SmoothL1Loss
        torch.nn.SmoothL1Loss = Capture
        try:
            if path == "memory":
                memory = ReplayMemory(100)
                for s, a, r in zip(states, actions, rewards):
                    memory.push(s, a, s, r, True)
                # train_time 10 (StepLR(step_size=train_time // 10) needs >= 10), lr 0: only the first batch is read
                net.train_model_memory(target, total_train_time=0, train_time=10, train_batch=8, gamma=0.99,
                                       memory=memory, optimizer=torch.optim.AdamW, lr=0.0)
            else:
                loader = DataLoader(ExampleSet(np.stack(states), actions, rewards), batch_size=8, shuffle=False)
                net.train_model_loader(target, total_train_time=0, train_time=1, gamma=0.99, loader=loader,
                                       optimizer=torch.optim.AdamW, lr=0.0)
        finally:
            torch.nn.SmoothL1Loss = original
        q, tgt, loss = captured[0]
        batch_states = pre[0]
        order = [next(i for i, s in enumerate(states) if torch.equal(torch.from_numpy(s), batch_states[k]))
                 for k in range(len(batch_states))]
        reward_in_order = torch.tensor([rewards[i] for i in order]).view(-1, 1)
        wrong = (tgt - reward_in_order).abs().view(-1) > 1e-7
        correct_loss = torch.nn.functional.smooth_l1_loss(q, reward_in_order, reduction="none").mean()
        results[path] = {"targets": tgt.view(-1).tolist(), "rewards": reward_in_order.view(-1).tolist(),
                         "rows_with_bootstrap": [int(i) for i in torch.nonzero(wrong).view(-1)],
                         "loss": float(loss.mean()), "loss_with_terminal_targets": float(correct_loss)}
    return results


def reference(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(VENDOR)  # DQNSymbolicBase / DQNRelational* read Utils/model and Utils/target_class from the cwd
    import torchvision
    from world_model.baselines.phyq.vendor.LearningAgents.RLNetwork import DQNRelationalImage as image_mod
    from world_model.baselines.phyq.vendor.LearningAgents.RLNetwork import DQNRelationalSymbolic as symbolic_mod
    classes = {"DQNRelationalSymbolic": symbolic_mod.DQNRelationalSymbolic,
               "DQNRelationalImage": image_mod.DQNRelationalImage}
    torch.set_num_threads(1)
    meta = {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
            "python": sys.version.split()[0], "written_at": utc_now(), "vendor_json": sha256_of(VENDOR / "VENDOR.json"),
            "vendor_files": {p: sha256_of(VENDOR / p) for p in (
                "LearningAgents/RLNetwork/DQNBase.py", "LearningAgents/RLNetwork/DQNSymbolicBase.py",
                "LearningAgents/RLNetwork/DQNRelationalSymbolic.py", "LearningAgents/RLNetwork/DQNRelationalImage.py",
                "StateReader/SymbolicStateReader.py", "Utils/model", "Utils/target_class")},
            "references": {}}
    for arm, cls_name in ARMS.items():
        cls = classes[cls_name]
        torch.manual_seed(SEED_INIT)
        network = cls(h=120, w=160, outputs=180, if_save_local=True, writer=DummyWriter(), device="cpu")
        init = clone_state(network)
        per_member, states, actions, frames = official_inputs(cls_name, network)
        network.eval()
        with torch.no_grad():
            q_eval = network(states.float())
        network.load_state_dict(init)
        steps = official_train(cls, network, states, actions, LR, outer_calls=2)
        network.load_state_dict(init)
        steps_lr0 = official_train(cls, network, states, actions, 0.0, outer_calls=1)
        payload = {"arm": arm, "class": cls_name, "init": init, "inputs": per_member, "frames": frames,
                   "states": states, "actions": actions, "rewards": REWARDS, "examples": examples(),
                   "q_eval": q_eval, "steps": steps, "steps_lr0": steps_lr0,
                   "parameters": sum(p.numel() for p in network.parameters() if p.requires_grad)}
        path = out_dir / f"reference-{arm}.pt"
        torch.save(payload, path)
        meta["references"][arm] = sha256_of(path)
        print(f"{arm}: {len(steps)} + {len(steps_lr0)} official updates recorded -> {path}")
    meta["upstream_bug"] = upstream_bug_demo()
    (out_dir / "reference-meta.json").write_text(json.dumps(meta, indent=1) + "\n")


# ---------------------------------------------------------------------------
# adapter side (novphy)
# ---------------------------------------------------------------------------

def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-12)


def global_norm(grads):
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads.values()])))


def grad_check(mine, official):
    norm = global_norm(official)
    floor = TOL["grad_floor_of_global_norm"] * norm
    worst_rel, worst_abs = 0.0, 0.0
    for name, ref in official.items():
        diff = float(torch.linalg.vector_norm(mine[name] - ref))
        size = float(torch.linalg.vector_norm(ref))
        if size > floor:
            worst_rel = max(worst_rel, diff / size)
        else:
            worst_abs = max(worst_abs, diff)
    return {"global_norm": global_norm(mine), "official_global_norm": norm,
            "global_norm_relative_delta": rel(global_norm(mine), norm), "tensor_max_relative_l2": worst_rel,
            "small_tensor_max_abs_l2": worst_abs, "floor": floor,
            "pass": rel(global_norm(mine), norm) <= TOL["grad_norm_relative"]
            and worst_rel <= TOL["grad_tensor_relative_l2"] and worst_abs <= floor}


def allclose_block(mine, official):
    worst, ok = 0.0, True
    for name, ref in official.items():
        mine_t = mine[name]
        if ref.dtype in (torch.int64, torch.long):
            ok &= bool(torch.equal(mine_t, ref))
            continue
        atol = TOL["buffer_atol_of_max"] * float(ref.abs().max()) if ref.numel() else 0.0
        ok &= bool(torch.allclose(mine_t, ref, rtol=TOL["buffer_rtol"], atol=atol))
        worst = max(worst, float((mine_t - ref).abs().max()) if ref.numel() else 0.0)
    return {"max_abs_delta": worst, "pass": ok}


def adapter_steps(nv, model, states, actions, lr, updates, resets):
    pre, q_out = {}, {}
    for name, param in model.named_parameters():
        param.register_hook(lambda g, name=name: pre.__setitem__(name, g.detach().clone()))
    model.register_forward_hook(lambda m, i, o: q_out.__setitem__("q", o.detach().clone()) if m.training else None)
    data = nv.PhyqData(inputs=states, obs=torch.arange(len(actions)), actions=torch.tensor(actions),
                       rewards=torch.tensor(REWARDS, dtype=torch.float32),
                       probabilities=np.full(len(actions), 1.0 / len(actions)), levels=tuple(MEMBERS))
    optimizer, scheduler = nv.make_optimizer(model, updates, lr, resets=resets)
    names = {id(p): n for n, p in model.named_parameters()}
    steps = []
    for b in range(updates):
        indices = np.arange(BATCH) + BATCH * (b % 2)
        out = nv.runner_train_step(model, optimizer, scheduler, data, indices)
        steps.append({"loss": out["loss"], "q": q_out["q"], "grads_pre": dict(pre),
                      "grads_post": {n: p.grad.detach().clone() for n, p in model.named_parameters()},
                      "lr": optimizer.param_groups[0]["lr"],
                      "params": {n: p.detach().clone() for n, p in model.named_parameters()},
                      "buffers": {n: b_.detach().clone() for n, b_ in model.named_buffers()},
                      "adam": {names[id(p)]: {"exp_avg": s["exp_avg"].clone(), "exp_avg_sq": s["exp_avg_sq"].clone(),
                                              "step": float(s["step"])} for p, s in optimizer.state.items()}})
    return steps


def compare_steps(mine_steps, official_steps):
    out, ok = [], len(mine_steps) == len(official_steps)
    for mine, ref in zip(mine_steps, official_steps):
        lr = ref["lr"]
        bound = TOL["param_max_abs_over_lr"] * lr + TOL["param_close_abs"]
        deltas = torch.cat([(mine["params"][n] - p).abs().view(-1) for n, p in ref["params"].items()])
        adam_ok, adam_worst, step_counts = True, 0.0, set()
        for name, state in ref["adam"].items():
            for key in ("exp_avg", "exp_avg_sq"):
                block = allclose_block({key: mine["adam"][name][key]}, {key: state[key]})
                adam_ok &= block["pass"]
                adam_worst = max(adam_worst, block["max_abs_delta"])
            adam_ok &= mine["adam"][name]["step"] == state["step"]
            step_counts.add(state["step"])
        entry = {
            "loss": float(mine["loss"]), "official_loss": float(ref["loss"]),
            "loss_relative_delta": rel(mine["loss"], ref["loss"]),
            "q_max_abs_delta": float((mine["q"] - ref["q"]).abs().max()),
            "grads_pre_clamp": grad_check(mine["grads_pre"], ref["grads_pre"]),
            "grads_post_clamp": grad_check(mine["grads_post"], ref["grads_post"]),
            "lr": mine["lr"], "official_lr": lr,
            "param_close_share": float((deltas <= TOL["param_close_abs"]).float().mean()),
            "param_max_abs_delta": float(deltas.max()), "param_bound": bound,
            "buffers": allclose_block(mine["buffers"], ref["buffers"]),
            "adamw_moments_max_abs_delta": adam_worst, "adamw_steps": sorted(step_counts), "adamw_pass": adam_ok,
        }
        entry["pass"] = bool(entry["loss_relative_delta"] <= TOL["loss_relative"]
                             and entry["q_max_abs_delta"] <= TOL["q_max_abs"]
                             and entry["grads_pre_clamp"]["pass"] and entry["grads_post_clamp"]["pass"]
                             and entry["lr"] == lr and entry["param_close_share"] >= TOL["param_close_share"]
                             and entry["param_max_abs_delta"] <= bound and entry["buffers"]["pass"] and adam_ok)
        ok &= entry["pass"]
        out.append(entry)
    return out, bool(ok)


def compare_arm(nv, eb, ref):
    arm = ref["arm"]
    fit_keys = eb.fit_anchors()["keys"]
    anchor_frames = eb.fit_anchor_frames() if arm == "A3-img" else None
    inputs = {}
    for member in MEMBERS:
        if fit_keys[member] != f"{member}-c00":
            raise ValueError(f"{member}: anchor key {fit_keys[member]} differs")
        attempt = CAMPAIGN_FIT / fit_keys[member]
        if arm == "A3-sym":
            mine = nv.symbolic_state(attempt)
            inputs[member] = {"identical": bool(torch.equal(mine, ref["inputs"][member])),
                              "occupied_cells": int(mine.sum())}
        else:
            frame_equal = bool(torch.equal(anchor_frames[member], ref["frames"][member]))
            mine = nv.image_state(anchor_frames[member])
            inputs[member] = {"frame_equals_eb_fit_anchor_frame": frame_equal,
                              "max_abs_delta": float((mine - ref["inputs"][member]).abs().max()),
                              "identical": bool(torch.equal(mine, ref["inputs"][member]))}
    torch.manual_seed(SEED_INIT)
    model = nv.build_model(arm)
    init_mine = clone_state(model)
    init_identical = set(init_mine) == set(ref["init"]) and all(torch.equal(init_mine[k], v)
                                                                for k, v in ref["init"].items())
    missing, unexpected = model.load_state_dict(ref["init"], strict=True)
    states = ref["states"]
    model.eval()
    with torch.no_grad():
        q_eval = model(states.float())
    eval_delta = float((q_eval - ref["q_eval"]).abs().max())
    # score_actions path on the first anchor (Q of its four launches)
    first = MEMBERS[0]
    observation = (nv.symbolic_state(CAMPAIGN_FIT / f"{first}-c00") if arm == "A3-sym" else anchor_frames[first])
    launches = torch.tensor([[GRID[o][0] / 480., GRID[o][1] / 480., 1.0, 0.0] for _, o in ref["examples"][:4]])
    scored = nv.score_actions(model, observation, launches)
    score_delta = float((scored - ref["q_eval"][0, torch.tensor(ref["actions"][:4])]).abs().max())

    model.load_state_dict(ref["init"])
    steps, steps_ok = compare_steps(adapter_steps(nv, model, states, ref["actions"], LR, 4, [2]), ref["steps"])
    model = nv.build_model(arm)
    model.load_state_dict(ref["init"])
    steps0, steps0_ok = compare_steps(adapter_steps(nv, model, states, ref["actions"], 0.0, 2, []), ref["steps_lr0"])
    inputs_ok = all(v["identical"] for v in inputs.values()) and all(
        v.get("frame_equals_eb_fit_anchor_frame", True) for v in inputs.values())
    passed = bool(inputs_ok and eval_delta <= TOL["q_max_abs"] and score_delta <= TOL["q_max_abs"]
                  and steps_ok and steps0_ok)
    return {
        "configuration": f"official: {ref['class']} (h 120, w 160, 180 outputs, ImageNet ResNet-18 trunk), "
                         "DQNBase.train_model_loader (SmoothL1, clamp +-1, terminal targets), AdamW 3e-4 fresh per "
                         "outer step, batch 16 x 2 per outer step (G-port size; native batch 32)",
        "members": MEMBERS, "examples": len(ref["actions"]), "actions": ref["actions"],
        "trainable_parameters": nv.trainable_parameters(model), "official_trainable_parameters": ref["parameters"],
        "init_identical_under_seed": bool(init_identical), "state_dict_strict_load": not missing and not unexpected,
        "inputs": inputs, "inputs_pass": bool(inputs_ok),
        "eval_q_max_abs_delta": eval_delta, "score_actions_max_abs_delta": score_delta,
        "steps_lr_3e-4": steps, "steps_lr_0": steps0, "pass": passed,
    }


def compare(out_dir, write=True):
    out_dir = Path(out_dir)
    from scripts import external_baselines as eb
    from world_model.baselines.phyq import novphy as nv
    torch.set_num_threads(1)
    meta = json.loads((out_dir / "reference-meta.json").read_text())
    arms = {}
    for arm in ARMS:
        path = out_dir / f"reference-{arm}.pt"
        if sha256_of(path) != meta["references"][arm]:
            raise ValueError(f"{path} differs from reference-meta.json")
        arms[arm] = compare_arm(nv, eb, torch.load(path, map_location="cpu", weights_only=False))
    bug = meta["upstream_bug"]
    result = {
        "schema": "issue_127_gport_v1", "issue": 128, "arm_family": "A3 PhyQ-RelDQN", "checked_at": utc_now(),
        "tolerance": TOL,
        "official": {k: meta[k] for k in ("torch", "torchvision", "numpy", "python", "written_at", "vendor_json",
                                          "vendor_files", "references")}
        | {"upstream": "https://github.com/phy-q/benchmark", "commit": "514491c5e77dbacdd61066d11139434f982e3d9e",
           "runtime_compat": "np.float = float set around SymbolicStateReader.get_symbolic_image (VENDOR.json "
                             "compatibility); torchvision 0.28 resnet18(pretrained=True) = IMAGENET1K_V1, "
                             "tensor-identical to torchvision 0.9.1 resnet18-5c106cde.pth"},
        "adapter": {"module": "world_model/baselines/phyq/novphy.py",
                    "module_sha256": sha256_of(ROOT / "world_model/baselines/phyq/novphy.py"),
                    "imagenet_weights": {"path": "data/external/phyq/resnet18-5c106cde.pth", "url": nv.IMAGENET_URL,
                                         "sha256": "sha256:" + nv.IMAGENET_SHA256},
                    "resnet_definition": "world_model/baselines/phyre_dqn/vendor/torchvision/models/resnet.py "
                                         "(torchvision v0.4.0, BSD-3)",
                    "torch": torch.__version__, "python": sys.version.split()[0]},
        "arms": arms,
        "upstream_bug": {
            "location": "LearningAgents/RLNetwork/DQNBase.py train_model_memory: is_done_batch = "
                        "torch.Tensor(batch.is_done).long(); next_state_values[is_done_batch] = 0 (integer indexing: "
                        "zeroes rows 0 and 1, not the terminal rows)",
            "official_training_path": "TrainLearningAgent.py runs with --if_save_local True (default; "
                                      "TrainLearningAgent.sh passes no flag) -> train_model_loader, where is_done is a "
                                      "bool tensor (pandas bool column, default_collate) and the mask is correct",
            "toy_demo": bug,
        },
    }
    result["pass"] = all(a["pass"] for a in arms.values())
    if write:
        (out_dir / "gport.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({arm: {"pass": a["pass"], "init_identical": a["init_identical_under_seed"],
                            "eval_q": a["eval_q_max_abs_delta"],
                            "loss_rel": max(s["loss_relative_delta"] for s in a["steps_lr_3e-4"]),
                            "param_max": max(s["param_max_abs_delta"] for s in a["steps_lr_3e-4"])}
                      for arm, a in arms.items()}, indent=1))
    print("pass", result["pass"])
    return result


if __name__ == "__main__":
    mode = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else OUT
    if mode == "reference":
        reference(target)
    elif mode == "compare":
        compare(target)
    else:
        raise SystemExit("usage: b127_phyq_port_reference.py reference|compare [out_dir]")
